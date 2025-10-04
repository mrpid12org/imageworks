from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatMessagePartImage(BaseModel):
    type: str
    image_url: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    content: Any  # str | list[dict]
    # We keep it permissive; validation for image parts done later.


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolSpec(BaseModel):
    type: str = Field("function")
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Any] = None  # Could be "auto" | {"type": ...}
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    user: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChoiceMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    # Legacy compatibility field; not emitted after normalization.
    function_call: Optional[Dict[str, Any]] = None


class ChatChoice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[ChatChoice]
    created: int
    model: str
    usage: Optional[Usage] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"
