from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"   
    LLAMA_3_1_8B = "llama-3.1-8b-instant"        
    GPT_OSS_120B = "openai/gpt-oss-120b"         


class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None)
    model: ModelName = Field(default=ModelName.LLAMA_3_3_70B)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

    class Config:
        from_attributes = True


class DeleteFileRequest(BaseModel):
    file_id: int