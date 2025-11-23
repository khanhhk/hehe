from pydantic import BaseModel, Field


class SearchArgs(BaseModel):
    """
    Input model for search or retrieval operations.

    Attributes:
        query (str): The user's search query or question.
        top_k (int): The number of top results to return.
        with_score (bool): Whether to include relevance scores in the results.
    """

    query: str = Field(
        default="What do beetles eat?",
        description="The user's search query or question.",
    )
    top_k: int = Field(
        default=3, description="The number of top search results to return."
    )
    with_score: bool = Field(
        default=False,
        description="Whether to include the relevance score of each result.",
    )
