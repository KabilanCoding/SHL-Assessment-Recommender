"""
Pydantic schemas for the SHL Assessment Recommender API.
Schema is non-negotiable — matches the automated evaluator spec exactly.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single turn in the conversation history."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Stateless request — caller sends full conversation history every turn."""
    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    """One SHL assessment from the catalog."""
    name: str
    url: str
    test_type: str  # K | P | S | A | B | C | D | E


class ChatResponse(BaseModel):
    """
    Response from the agent.
    - recommendations: [] when clarifying, refusing, or comparing.
    - end_of_conversation: True only when the agent considers task complete.
    """
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
