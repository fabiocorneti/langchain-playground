"""
Request/response models
"""
from pydantic import BaseModel, Field, validator


class Question(BaseModel):
    """
    A question
    """
    question: str = Field(description="The question", max_length=1000)

    @validator('question', always=True)
    @classmethod
    def clean_question(cls, value):
        """
        Question validator
        """
        value = value.replace("\r\n", "")
        assert value, "No question was asked"
        return value


class Metadata(BaseModel):
    """
    Document metadata
    """
    page: int = Field(description="Page number")
    source: str = Field(description="Source")
    title: str = Field(description="Title")


class Result(BaseModel):
    """
    Document result.
    """
    page_content: str = Field(description="Snippet")
    metadata: Metadata = Field(description="Metadata")
    score: float = Field(description="Score")


class Answer(BaseModel):
    """
    An answer
    """
    answer: str = Field(description="The answer from LLM if configured")
    results: list[Result] = Field(description="List of chunks for the question")
