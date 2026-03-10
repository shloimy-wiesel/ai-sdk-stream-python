# ai-sdk-stream-python

A Python package for AI SDK streaming utilities built with [FastAPI](https://fastapi.tiangolo.com/) and [Pydantic](https://docs.pydantic.dev/).

## Installation

```bash
pip install ai-sdk-stream-python
```

To run the server, you'll also need an ASGI server such as [uvicorn](https://www.uvicorn.org/):

```bash
pip install uvicorn
```

## Usage

```python
from ai_sdk_stream_python import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

Or run directly:

```bash
uvicorn ai_sdk_stream_python:app --reload
```

## Endpoints

- `GET /` — Returns package name and version.
- `GET /messages` — Returns a list of dummy messages.
