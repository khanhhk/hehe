import asyncio
import json
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Union
from uuid import UUID

import redis
from nemoguardrails import LLMRails

from src.config.settings import SETTINGS
from src.utils.logger import FrameworkLogger, get_logger

logger: FrameworkLogger = get_logger()


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder to convert UUIDs to strings."""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(self, obj)


class StandardCache:
    """
    Standard caching layer for both sync and async functions using Redis.

    Supports:
    - Transparent decorator-based cache control.
    - Key generation based on function metadata and arguments.
    - Optional response validation via Pydantic models.
    """

    def __init__(self) -> None:
        self.storage_uri = f"redis://{SETTINGS.REDIS_URI}"
        self.client = redis.Redis(
            host=self.storage_uri.split(":")[1].replace("//", ""),
            port=int(self.storage_uri.split(":")[2]),
        )

    def _cache_logic(
        self,
        func: Callable,
        args: Tuple,
        kwargs: dict,
    ) -> Tuple[Optional[str], Optional[Union[str, dict]]]:
        """
        Core logic to construct cache key and check Redis for a cached value.

        Returns:
            - ("hit", result): on cache hit
            - ("miss", key): on cache miss
            - (None, None): if Redis is unavailable
        """
        environment = SETTINGS.ENVIRONMENT
        module_name = func.__module__
        func_name = func.__qualname__

        # Strip 'self' and filter out LLMRails instances in args
        if args and hasattr(args[0], func.__name__):
            # It's a class method: remove 'self'
            args_to_serialize = tuple(
                arg for arg in args[1:] if not isinstance(arg, LLMRails)
            )
            class_name = args[0].__class__.__name__
            func_name = f"{class_name}.{func.__name__}"
        else:
            args_to_serialize = tuple(
                arg for arg in args if not isinstance(arg, LLMRails)
            )

        # Filter out LLMRails instances in kwargs
        kwargs_to_serialize = {
            k: v for k, v in kwargs.items() if not isinstance(v, LLMRails)
        }

        # Build cache key
        dumped_args = self.serialize(args_to_serialize)
        dumped_kwargs = self.serialize(kwargs_to_serialize)
        key = (
            f"mlops:{environment}:{module_name}:"
            + f"{func_name}:{dumped_args}:{dumped_kwargs}"
        )
        logger.info(f"Cached key: {key}")

        # Try to retrieve from cache
        try:
            cached_result = self.client.get(key)
            logger.info(f"Cache lookup: {'HIT' if cached_result else 'MISS'}")
        except Exception as e:
            logger.warning(f"Redis error accessing key {key}: {e}")
            return None, None

        if cached_result:
            return "hit", self.deserialize(cached_result)
        return "miss", key

    def cache(self, *, ttl: int = 60 * 60, validatedModel: Any = None) -> Callable:
        """
        Decorator for caching both sync and async functions.

        Args:
            ttl (int): Time-to-live for the cache entry in seconds.
            validatedModel (Optional[Any]): Pydantic model to validate cached response.
        """

        def inner(func):
            is_async = asyncio.iscoroutinefunction(func)

            if is_async:

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    cache_result, data = self._cache_logic(
                        func, args, kwargs, ttl, validatedModel, True
                    )

                    if cache_result is None:
                        return await func(*args, **kwargs)
                    elif cache_result == "hit":
                        return data
                    else:
                        result = await func(*args, **kwargs)
                        self._store_result(data, result, ttl, validatedModel)
                        return result

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    cache_result, data = self._cache_logic(
                        func, args, kwargs, ttl, validatedModel, False
                    )

                    if cache_result is None:
                        return func(*args, **kwargs)
                    elif cache_result == "hit":
                        return data
                    else:
                        result = func(*args, **kwargs)
                        self._store_result(data, result, ttl, validatedModel)
                        return result

                return sync_wrapper

        return inner

    def _store_result(
        self,
        key: str,
        result: Any,
        ttl: int,
        validatedModel: Optional[Any] = None,
    ) -> None:
        """
        Store function result in Redis with optional validation.

        Supports:
        - Pydantic model serialization (single or list).
        - Fallback to raw JSON serialization.
        """
        try:
            # Handle Pydantic models
            if hasattr(result, "model_dump"):
                data_to_serialize = result.model_dump()
            elif (
                isinstance(result, list) and result and hasattr(result[0], "model_dump")
            ):
                data_to_serialize = [r.model_dump() for r in result]
            else:
                data_to_serialize = result

            serialized_result = self.serialize(data_to_serialize)
        except TypeError as e:
            logger.warning(f"Serialization failed for key: {key}, error: {e}")
            return

        # Optional validation step before storing
        if validatedModel:
            try:
                validatedModel(**self.deserialize(serialized_result))
            except Exception as e:
                logger.warning(f"Validation failed for key: {key}, error: {e}")
                return

        # Store in Redis
        self.set_key(key, serialized_result, ttl)
        logger.info(f"Cache STORED for key: {key}")

    def set_key(self, key: str, value: Any, ttl: int = 60 * 60) -> None:
        """
        Set a Redis key with TTL.

        Args:
            key (str): The Redis key.
            value (Any): The value to store (JSON serialized).
            ttl (int): Time to live in seconds.
        """
        self.client.set(key, value)
        self.client.expire(key, ttl)

    def remove_key(self, key: str) -> None:
        """
        Delete a key from Redis cache.

        Args:
            key (str): The key to remove.
        """
        self.client.delete(key)

    def serialize(self, value: Any) -> str:
        """
        Serialize Python object to JSON string.

        Args:
            value (Any): The object to serialize.

        Returns:
            str: JSON representation.
        """
        return json.dumps(value, cls=UUIDEncoder, sort_keys=True)

    def deserialize(self, value: Union[str, bytes]) -> dict:
        """
        Deserialize JSON string from Redis.

        Args:
            value (str | bytes): Raw JSON string.

        Returns:
            dict: Parsed object.
        """
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    def list_keys(self, pattern: str = f"mlops:{SETTINGS.ENVIRONMENT}:*") -> list:
        """
        List all Redis keys matching a pattern.

        Args:
            pattern (str): Pattern to match keys.

        Returns:
            list: Matching Redis keys.
        """
        return self.client.keys(pattern)


# Singleton instance
standard_cache = StandardCache()
