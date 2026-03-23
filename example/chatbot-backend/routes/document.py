from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from services import redis_store

router = APIRouter()


@router.get("/document")
async def get_document(id: str = Query(...)) -> JSONResponse:
    doc = await redis_store.get_document(id)
    if doc is None:
        return JSONResponse([], status_code=200)
    return JSONResponse([doc], status_code=200)
