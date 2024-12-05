from pydantic import BaseModel, Field
from typing import List, Optional

class Question(BaseModel):
    topic: str = Field(description="Topic of the questions that has been generated")
    question: str = Field(description="Interview question for the candidate")
    answer: str = Field(description="Suggested answer for the question")