from fastapi import FastAPI
from pydantic import BaseModel, Field
import httpx
import os
import time
import uuid

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "http://qdrant:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
QDRANT_KB_COLLECTION = os.getenv("QDRANT_KB_COLLECTION", "mem_kb")
QDRANT_CHAT_COLLECTION = os.getenv("QDRANT_CHAT_COLLECTION", "mem_chat")
KB_TOP_K = int(os.getenv("KB_TOP_K", "3"))
CHAT_TOP_K = int(os.getenv("CHAT_TOP_K", "5"))
CHAT_SCORE_THRESHOLD = float(os.getenv("CHAT_SCORE_THRESHOLD", "0.75"))
KB_SCORE_THRESHOLD = float(os.getenv("KB_SCORE_THRESHOLD", "0.75"))
CHAT_BASE_THRESHOLD = float(os.getenv("CHAT_BASE_THRESHOLD", "0.60"))
CHAT_GAP_THRESHOLD  = float(os.getenv("CHAT_GAP_THRESHOLD",  "0.05"))
CHAT_MARGIN         = float(os.getenv("CHAT_MARGIN",         "0.02"))


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


async def qdrant_scroll_chat(session_id: str, limit: int = 20):
    body = {
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
        "filter": {"must": [{"key": "session_id", "match": {"value": session_id}}]},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_CHAT_COLLECTION}/points/scroll",
            json=body,
        )
        r.raise_for_status()
        return r.json().get("result", {}).get("points", [])


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

    # 1) embedding 用户输入（只算向量，不写库）
    qvec = await ollama_embed(req.message)

    # 2) 先检索历史记忆（注意：此时还没把本次 user 写进去，所以不会“命中自己”）
    hits = await qdrant_search_chat(session_id, qvec, CHAT_TOP_K)

    # ---- Dynamic gating (research-y) ----
    top1 = hits[0].get("score") if len(hits) >= 1 else None
    top2 = hits[1].get("score") if len(hits) >= 2 else None

    top1_score = top1
    top2_score = top2

    dynamic_th = None
    gap_ok = False
    th_ok = False

    if top1 is not None:
        dynamic_th = max(CHAT_BASE_THRESHOLD, top1 - CHAT_MARGIN)
        th_ok = top1 >= dynamic_th

        if top2 is None:
            gap_ok = True
        else:
            gap_ok = (top1 - top2) >= CHAT_GAP_THRESHOLD

    memory_allowed = bool(th_ok and gap_ok)
    # ------------------------------------

    # 3) 如果不允许用记忆：只写入一次 user（可选，但推荐），然后直接拒绝
    if not memory_allowed:
        if req.message.strip():
            await qdrant_upsert_chat(session_id, "user", req.message, qvec)

        return {
            "session_id": session_id,
            "model": OLLAMA_MODEL,
            "reply": "...",
            "refusal": True,
            "memory_used": [],
            "retrieval": {
                "query": req.message,
                "top_k": CHAT_TOP_K,
                "base_threshold": CHAT_BASE_THRESHOLD,
                "gap_threshold": CHAT_GAP_THRESHOLD,
                "margin": CHAT_MARGIN,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "dynamic_threshold": dynamic_th,
                "th_ok": th_ok,
                "gap_ok": gap_ok,
                "hit_count_raw": len(hits),
                "hit_count_used": 0,
            },
        }

    # 4) 通过 gating：把 hits 里的 payload text 取出来当 memories
    memories = []
    for h in hits:
        p = h.get("payload") or {}
        t = p.get("text")
        if t:
            memories.append(t)

    # 5) 拼 system prompt
    system_text = req.system or "你是一个有长期记忆的助手。"
    if memories:
        memory_block = "以下是与当前问题相关的历史记忆（可能不完全准确）：\n" + "\n".join(
            [f"- {m}" for m in memories[:CHAT_TOP_K]]
        )
        system_text = system_text + "\n\n" + memory_block

    # 6) 调用 LLM
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

    # 7) 最后统一写入记忆：user + assistant（都只写一次）
    if req.message.strip():
        await qdrant_upsert_chat(session_id, "user", req.message, qvec)

    if reply.strip():
        avec = await ollama_embed(reply)
        await qdrant_upsert_chat(session_id, "assistant", reply, avec)

    return {
        "session_id": session_id,
        "model": OLLAMA_MODEL,
        "reply": reply,
        "refusal": False,
        "memory_used": memories[:CHAT_TOP_K],
        "retrieval": {
            "query": req.message,
            "top_k": CHAT_TOP_K,
            "base_threshold": CHAT_BASE_THRESHOLD,
            "gap_threshold": CHAT_GAP_THRESHOLD,
            "margin": CHAT_MARGIN,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "dynamic_threshold": dynamic_th,
            "th_ok": th_ok,
            "gap_ok": gap_ok,
            "hit_count_raw": len(hits),
            "hit_count_used": len(memories[:CHAT_TOP_K]),
        }
    }


@app.post("/chat_mem_strict")
async def chat_mem_strict(req: ChatReq):
    session_id = req.session_id or "demo"

    # 1) embedding 用户输入
    qvec = await ollama_embed(req.message)

    # 2) 检索历史记忆
    hits = await qdrant_search_chat(session_id, qvec, CHAT_TOP_K)

    # 3) 阈值过滤
    memories = []
    kept = []  # 用于返回调试信息：point_id/score
    for h in hits:
        score = h.get("score")
        p = h.get("payload") or {}
        t = p.get("text")
        pid = h.get("id")

        if t and (score is not None) and (score >= CHAT_SCORE_THRESHOLD):
            memories.append(t)
            kept.append({"point_id": pid, "score": score})

    # 4) HARD REFUSAL：没有任何通过阈值的记忆 -> 不调用 LLM
    if not memories:
        return {
            "session_id": session_id,
            "model": OLLAMA_MODEL,
            "reply": "...",
            "refusal": True,
            "memory_used": [],
            "retrieval": {
                "query": req.message,
                "top_k": CHAT_TOP_K,
                "score_threshold": CHAT_SCORE_THRESHOLD,
                "hit_count_raw": len(hits),
                "hit_count_filtered": 0,
            }
        }

    # 5) 拼 system prompt
    system_text = req.system or "你是一个有长期记忆的助手。"
    memory_block = "以下是与当前问题相关的历史记忆（可能不完全准确）：\n" + "\n".join(
        [f"- {m}" for m in memories[:CHAT_TOP_K]]
    )
    system_text = system_text + "\n\n" + memory_block

    # 6) 调用 LLM
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

    # 7) 写入记忆：user + assistant（和你现在的 chat_mem 一样）
    if req.message.strip():
        await qdrant_upsert_chat(session_id, "user", req.message, qvec)

    if reply.strip():
        avec = await ollama_embed(reply)
        await qdrant_upsert_chat(session_id, "assistant", reply, avec)

    return {
        "session_id": session_id,
        "model": OLLAMA_MODEL,
        "reply": reply,
        "refusal": False,
        "memory_used": memories[:CHAT_TOP_K],
        "retrieval": {
            "query": req.message,
            "top_k": CHAT_TOP_K,
            "score_threshold": CHAT_SCORE_THRESHOLD,
            "hit_count_raw": len(hits),
            "hit_count_filtered": len(memories[:CHAT_TOP_K]),
            "kept": kept,
        }
}

@app.get("/debug/chat_mem/{session_id}")
async def debug_chat_mem(session_id: str, limit: int = 20):
    points = await qdrant_scroll_chat(session_id, limit=limit)
    items = []
    for p in points:
        payload = p.get("payload") or {}
        items.append({
            "point_id": p.get("id"),
            "role": payload.get("role"),
            "text": payload.get("text"),
            "ts": payload.get("ts"),
        })
    return {
        "session_id": session_id,
        "collection": QDRANT_CHAT_COLLECTION,
        "count": len(items),
        "items": items,
    }
class DebugChatSearchReq(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int | None = None

@app.post("/debug/chat_mem_search")
async def debug_chat_mem_search(req: DebugChatSearchReq):
    top_k = req.top_k or CHAT_TOP_K
    qvec = await ollama_embed(req.query)
    hits = await qdrant_search_chat(req.session_id, qvec, top_k)
    top1_score = hits[0].score if hits else None
    top2_score = hits[1].score if len(hits) > 1 else None

    out = []
    for h in hits:
        p = h.get("payload") or {}
        out.append({
            "score": h.get("score"),
            "point_id": h.get("id"),
            "role": p.get("role"),
            "text": p.get("text"),
            "ts": p.get("ts"),
        })

    return {
        "session_id": req.session_id,
        "top_k": top_k,
        "hits": out,
    }

class ChatKBReq(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    system: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    score_threshold: float | None = None

class KBUpsertReq(BaseModel):
    text: str = Field(..., min_length=1)
    id: str | None = None


async def kb_search(query: str, top_k: int):
    # 1. 生成 embedding
    qvec = await ollama_embed(query)

    # 2. 搜索 Qdrant
    body = {
        "vector": qvec,
        "limit": top_k,
        "with_payload": True
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}/points/search",
            json=body
        )
        r.raise_for_status()
        return r.json()["result"]
async def kb_upsert(text: str, point_id: str | None = None):
    vec = await ollama_embed(text)
    if point_id is None:
        point_id = str(uuid.uuid4())

    body = {
        "points": [
            {
                "id": point_id,
                "vector": vec,
                "payload": {"text": text}
            }
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.put(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}/points",
            json=body,
        )
        r.raise_for_status()
        return {"id": point_id, "result": r.json()}

@app.post("/kb_upsert")
async def kb_upsert_api(req: KBUpsertReq):
    return await kb_upsert(req.text, req.id)

@app.post("/chat_kb")
async def chat_kb(req: ChatKBReq):
    top_k = req.top_k if req.top_k else KB_TOP_K
    score_th = req.score_threshold if req.score_threshold is not None else KB_SCORE_THRESHOLD

    hits = await kb_search(req.message, top_k)
    raw_hit_count = len(hits)

    contexts = []
    sources = []  # 这里我们改名叫 sources（返回给用户的，可追溯）
    cite_no = 0   # 引用编号：1,2,3...（只给“通过阈值”的命中编号，避免中间断号）

    for h in hits:
        point_id = h.get("id")  # ✅ 真实 Qdrant point id（uuid 或数字）
        payload = h.get("payload") or {}
        txt = payload.get("text") or ""
        score = h.get("score")

        if (txt is not None) and (txt.strip() != "") and (score is not None) and (score >= score_th):
            cite_no += 1
            contexts.append(f"[{cite_no}] {txt}")
            sources.append({
                "cite": cite_no,  # ✅ 给 LLM 引用的编号
                "point_id": point_id,  # ✅ 真正可追溯的 Qdrant id
                "score": score,
                "collection": QDRANT_KB_COLLECTION,  # ✅ 追溯到哪个 collection
                "text": txt
            })
    filtered_hit_count = len(sources)

    context_block = "\n".join(contexts) if contexts else "(知识库未命中)"
    # HARD REFUSAL: after threshold filtering, no usable KB context -> do not call LLM
    if not sources:
        return {
            "model": OLLAMA_MODEL,
            "reply": "...",
            "sources": [],
            "refusal": True,
            "retrieval": {
                "query": req.message,
                "top_k": top_k,
                "score_threshold": score_th,
                "hit_count_raw": raw_hit_count,
                "hit_count_filtered": 0
            }
        }

    system_prompt = (
        "你是一个检索增强助手。请基于知识库回答问题，并在使用知识时用 [编号] 标注引用。\n\n"
        f"【知识库】\n{context_block}\n\n"
        "【用户问题】"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ]
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    return {
        "model": OLLAMA_MODEL,
        "reply": data.get("message", {}).get("content", ""),
        "sources": sources,
        "retrieval": {
            "query": req.message,
            "top_k": top_k,
            "score_threshold": score_th,
            "hit_count_raw": raw_hit_count,
            "hit_count_filtered": filtered_hit_count
        }
    }
