from pydantic import BaseModel
from typing import Any

class ParameterDef(BaseModel):
    type: str  # "number", "string", "boolean"


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDef]
    returns: ParameterDef


class Prompt(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
