from fastapi import FastAPI
from pydantic import BaseModel, Field
import httpx
import os
import time
import uuid

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "http://qdrant:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
QDRANT_CHAT_COLLECTION = os.getenv("QDRANT_CHAT_COLLECTION", "mem_chat")
CHAT_TOP_K = int(os.getenv("CHAT_TOP_K", "6"))

app = FastAPI(title="Agent API", version="0.1.0")

class ChatReq(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    system: str | None = None
    temperature: float | None = None
async def ollama_embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]

async def qdrant_upsert_chat(session_id: str, role: str, text: str, vector: list[float]):
    point_id = str(uuid.uuid4())
    payload = {
        "type": "chat",
        "session_id": session_id,
        "role": role,
        "text": text,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    body = {"points": [{"id": point_id, "vector": vector, "payload": payload}]}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.put(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_CHAT_COLLECTION}/points",
            json=body,
        )
        r.raise_for_status()
        return r.json()

async def qdrant_search_chat(session_id: str, query_vector: list[float], limit: int):
    body = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
        "filter": {"must": [{"key": "session_id", "match": {"value": session_id}}]},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_CHAT_COLLECTION}/points/search",
            json=body,
        )
        r.raise_for_status()
        return r.json().get("result", [])

@app.get("/health")
async def health():
    return {"ok": True, "model": OLLAMA_MODEL}

@app.post("/chat")
async def chat(req: ChatReq):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": []
    }
    if req.system:
        payload["messages"].append({"role": "system", "content": req.system})
    payload["messages"].append({"role": "user", "content": req.message})

    if req.temperature is not None:
        payload["options"] = {"temperature": req.temperature}

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    # Ollama /api/chat returns {"message": {"role": "...", "content": "..."}, ...}
    return {
        "model": OLLAMA_MODEL,
        "reply": data.get("message", {}).get("content", ""),
        "raw": data,
    }


@app.get("/test")
async def test():
    return {"ok": True}

@app.post("/chat_mem")
async def chat_mem(req: ChatReq):
    session_id = req.session_id or "demo"

    # 1) embedding 用户输入
    qvec = await ollama_embed(req.message)

    # 2) 检索历史记忆
    hits = await qdrant_search_chat(session_id, qvec, CHAT_TOP_K)
    memories = []
    for h in hits:
        p = h.get("payload") or {}
        t = p.get("text")
        if t:
            memories.append(t)

    # 3) 把记忆拼进 system prompt（最简单做法）
    system_text = req.system or "你是一个有长期记忆的助手。"
    if memories:
        memory_block = "以下是与当前问题相关的历史记忆（可能不完全准确）：\n" + "\n".join(
            [f"- {m}" for m in memories[:CHAT_TOP_K]]
        )
        system_text = system_text + "\n\n" + memory_block

    # 4) 调用 LLM
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": req.message},
        ],
    }
    if req.temperature is not None:
        payload["options"] = {"temperature": req.temperature}

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    reply = data.get("message", {}).get("content", "")

    # 5) 写入记忆：user + assistant
    await qdrant_upsert_chat(session_id, "user", req.message, qvec)

    return {
        "session_id": session_id,
        "model": OLLAMA_MODEL,
        "reply": reply,
        "memory_used": memories[:CHAT_TOP_K],
    }
