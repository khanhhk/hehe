import inspect
import json
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

from langchain_core.outputs import Generation
from langchain_redis import RedisSemanticCache

from src.infrastructure.embeddings.embeddings import EmbeddingService
from src.utils.logger import FrameworkLogger, get_logger
from src.utils.text_processing import build_context

logger: FrameworkLogger = get_logger()


class SemanticCacheLLMs:
    """
    Caching layer for LLM responses using Redis + semantic similarity search.

    Supports both:
    - SSE (streaming) mode: yields tokens/chunks
    - REST mode: returns final response

    Cached based on semantic similarity of the input "context string".
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6378",
        *,
        embeddings: Optional[Any] = None,
        distance_threshold: float = 0.2,
        ttl: int = 20,
    ):
        """
        Initialize the semantic cache.

        Args:
            redis_url (str): Redis connection URI.
            embeddings (Optional[Any]): Embedding model.
            distance_threshold (float): Max distance for cache hit.
            ttl (int): Time-to-live (in minutes) for cache entries.
        """
        self._cache = RedisSemanticCache(
            embeddings=embeddings or EmbeddingService(),
            redis_url=redis_url,
            distance_threshold=distance_threshold,
            ttl=ttl,
            name="llm_cache",
            prefix="llmcache",
        )
        logger.info(
            f"SemanticCacheLLMs init (threshold={distance_threshold}, ttl={ttl})"
        )

    def _get_context_str(self, **kwargs: Any) -> Optional[str]:
        """
        Extract the semantic cache key from arguments.

        Priority:
        1. Use `messages` (chat format → build_context)
        2. Fallback to `question` (string)

        Returns:
            Optional[str]: String used as cache key.
        """
        question = kwargs.get("question")
        messages = kwargs.get("messages")
        if messages:  # post-cache
            return build_context(messages)
        return question  # pre-cache

    async def _handle_sse_cache_hit(self, hit: Generation) -> AsyncGenerator[str, None]:
        """
        Stream cached SSE response chunk-by-chunk.

        Args:
            hit (Generation): Cached response generation.

        Yields:
            str: JSON string chunks with `\n\n` delimiter.
        """
        try:
            cached_data = json.loads(hit.text)
            response = cached_data.get("response", "")
            if isinstance(response, str):
                words = response.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    yield f"{json.dumps(chunk)}\n\n"
        except (json.JSONDecodeError, KeyError):
            yield f'{json.dumps("Error loading from cache")}\n\n'

    async def _execute_and_cache_sse(
        self,
        func: Callable[..., AsyncGenerator[str, None]],
        namespace: str,
        context_str: str,
        *args,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Execute streaming function and cache its full output.

        Args:
            func: The SSE generator function to wrap.
            namespace: Cache namespace.
            context_str: Key for cache.
        """
        full_response = ""
        async for chunk in func(*args, **kwargs):
            clean_chunk = chunk.replace("\n\n", "")
            try:
                # Try to decode JSON if it looks like JSON (starts and ends with quotes)
                if clean_chunk.strip().startswith('"') and clean_chunk.strip().endswith(
                    '"'
                ):
                    decoded_chunk = json.loads(clean_chunk)
                    full_response += decoded_chunk
                else:
                    full_response += clean_chunk
            except json.JSONDecodeError:
                full_response += clean_chunk
            yield chunk

        cache_data = {
            "type": "sse_response",
            "response": full_response.strip(),
        }
        self._cache.update(
            context_str,
            namespace,
            [Generation(text=json.dumps(cache_data))],
        )
        logger.info(f"SSE Cache-miss [{namespace}]: {context_str}")

    def _handle_rest_cache_hit(self, hit: Generation) -> Any:
        """
        Return REST cached response.

        Args:
            hit: Cached result from RedisSemanticCache.
        """
        cached_data = json.loads(hit.text)
        return cached_data["response"]

    async def _execute_and_cache_rest(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        namespace: str,
        context_str: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute REST function and cache its result.

        Args:
            func: Awaitable function returning result.
            namespace: Cache namespace.
            context_str: Cache key.
        """
        result = await func(*args, **kwargs)
        cache_data = {"type": "rest_response", "response": result}
        self._cache.update(
            context_str,
            namespace,
            [Generation(text=json.dumps(cache_data))],
        )
        logger.debug(f"Cache-miss → stored [{namespace}]: {context_str}")
        return result

    def cache(self, *, namespace: str) -> Callable:
        """
        Decorator to apply semantic caching to a function.

        Args:
            namespace (str): Semantic cache namespace (e.g., "search", "summary").

        Returns:
            Decorated function with semantic caching logic applied.
        """

        def inner(func: Callable):
            if inspect.isasyncgenfunction(func):  # SSE (async generator)

                @wraps(func)
                async def sse_wrapper(*args, **kwargs):
                    context_str = self._get_context_str(**kwargs)
                    hits = self._cache.lookup(context_str, namespace)  # type: ignore

                    if hits:
                        logger.info(f"SSE Cache-hit [{namespace}]: {context_str}")
                        async for chunk in self._handle_sse_cache_hit(hits[0]):
                            yield chunk
                    else:
                        async for chunk in self._execute_and_cache_sse(
                            func, namespace, context_str, *args, **kwargs
                        ):
                            yield chunk

                return sse_wrapper
            else:  # Coroutine (REST)

                @wraps(func)
                async def rest_wrapper(*args, **kwargs):
                    context_str = self._get_context_str(**kwargs)
                    hits = self._cache.lookup(context_str, namespace)  # type: ignore

                    if hits:
                        logger.info(f"REST Cache-hit [{namespace}]: {context_str}")
                        return self._handle_rest_cache_hit(hits[0])
                    else:
                        return await self._execute_and_cache_rest(
                            func, namespace, context_str, *args, **kwargs
                        )

                return rest_wrapper

        return inner


semantic_cache_llms = SemanticCacheLLMs()
