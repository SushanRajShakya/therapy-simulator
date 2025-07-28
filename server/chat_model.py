from pydantic import BaseModel
from typing import List, Dict, Optional


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: str
    end_session: bool = False  # Flag to indicate session ending


class ChatResponse(BaseModel):
    response: str
    session_id: str
    is_session_ended: bool = False  # Flag to indicate if session has ended
