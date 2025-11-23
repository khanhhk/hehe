from pydantic import BaseModel, Field


class UserInput(BaseModel):
    """
    Input model representing a user's query within a session context.

    Attributes:
        user_input (str): The question or prompt provided by the user.
        session_id (str): Unique identifier for the session.
        user_id (str): Unique identifier for the user.
    """

    user_input: str = Field(
        default="What do beetles eat?",
        description="The question or input provided by the user.",
    )
    session_id: str = Field(
        default="1", description="Unique identifier for the current session."
    )
    user_id: str = Field(default="1", description="Unique identifier for the user.")
