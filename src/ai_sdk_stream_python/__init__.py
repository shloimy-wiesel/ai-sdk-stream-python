from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ai-sdk-stream-python", version="0.1.0")


class Message(BaseModel):
    id: int
    content: str


_dummy_messages: list[Message] = [
    Message(id=1, content="Hello from ai-sdk-stream-python!"),
    Message(id=2, content="This is a dummy streaming message."),
]


@app.get("/")
def root() -> dict:
    return {"name": "ai-sdk-stream-python", "version": "0.1.0"}


@app.get("/messages", response_model=list[Message])
def get_messages() -> list[Message]:
    return _dummy_messages
