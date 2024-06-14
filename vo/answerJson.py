from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional


class Answer(BaseModel):
    content: str = Field(description="AI给出的回答")
    cite: List[str] = Field(description="AI的参考文献")


class OutJson(BaseModel):
    question: str = Field(description="用户输入的提问")
    answer: Optional[Answer] = Field(description="AI给出的回答")

