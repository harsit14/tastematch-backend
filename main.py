"""
TasteMatch Pilot — FastAPI Backend (Phase 4)
Single-file implementation — no local module imports needed.

Endpoints:
  GET  /health
  POST /auth/signup
  POST /auth/login
  POST /chat/session
  POST /chat/message
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import openai
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client
from google.oauth2 import service_account
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("tastematch")

# ─────────────────────────────────────────────
# Config from environment
# ─────────────────────────────────────────────
SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_ANON_KEY    = os.environ["SUPABASE_ANON_KEY"]

VERTEX_PROJECT  = os.environ.get("VERTEX_PROJECT", "tastematch-pilot")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
# Llama 3.3 70B is the smallest available Llama MaaS model on Vertex AI
VERTEX_MODEL    = os.environ.get("VERTEX_MODEL", "meta/llama-3.3-70b-instruct-maas")

CHROMA_HOST       = os.environ.get("CHROMA_HOST", "")
CHROMA_PORT       = int(os.environ.get("CHROMA_PORT", "8001"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "nutrition_data")

PORT = int(os.environ.get("PORT", "8000"))

# ─────────────────────────────────────────────
# Supabase admin client (service role — bypasses RLS)
# ─────────────────────────────────────────────
admin_supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─────────────────────────────────────────────
# Vertex AI — OpenAI-compatible client for Llama MaaS
#
# Llama models on Vertex AI do NOT use the GenerativeModel SDK.
# They use an OpenAI-compatible endpoint. We get a short-lived
# access token from the service account and use it as the API key.
#
# Supports two auth methods:
#   1. GOOGLE_CREDENTIALS_JSON — paste full JSON contents (Railway)
#   2. GOOGLE_APPLICATION_CREDENTIALS — path to JSON key file (local dev)
# ─────────────────────────────────────────────
def get_vertex_openai_client() -> openai.OpenAI:
    creds_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
    creds_file     = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    if creds_json_str:
        creds_dict  = json.loads(creds_json_str)
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        logger.info("Vertex AI: using GOOGLE_CREDENTIALS_JSON env var.")
    elif creds_file and os.path.exists(creds_file):
        credentials = service_account.Credentials.from_service_account_file(
            creds_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        logger.info(f"Vertex AI: using credentials file {creds_file}.")
    else:
        credentials, _ = google_auth_default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        logger.info("Vertex AI: using Application Default Credentials.")

    # Refresh to get a valid short-lived access token
    credentials.refresh(GoogleAuthRequest())

    return openai.OpenAI(
        base_url=(
            f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1beta1"
            f"/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/endpoints/openapi"
        ),
        api_key=credentials.token,
    )

# ─────────────────────────────────────────────
# Chroma (optional — stub until Phase 3 complete)
# ─────────────────────────────────────────────
chroma_collection = None

if CHROMA_HOST:
    try:
        import chromadb
        chroma_client     = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION)
        logger.info(f"Chroma connected: {CHROMA_HOST}:{CHROMA_PORT} / {CHROMA_COLLECTION}")
    except Exception as e:
        logger.warning(f"Chroma connection failed — RAG disabled. {e}")
else:
    logger.info("CHROMA_HOST not set — running without RAG (stub mode).")

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(title="TasteMatch API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Vercel URL before going live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    health_conditions: list[str]   = []
    dietary_preferences: list[str] = []

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class SessionRequest(BaseModel):
    title: Optional[str] = "New conversation"

class MessageRequest(BaseModel):
    session_id: str
    message: str

# ─────────────────────────────────────────────
# Auth helper — verify Supabase JWT
# ─────────────────────────────────────────────
async def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    token = authorization.split(" ", 1)[1]

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return resp.json()

# ─────────────────────────────────────────────
# RAG helper — query Chroma
# ─────────────────────────────────────────────
def get_rag_context(user_message: str, n_results: int = 3) -> str:
    if chroma_collection is None:
        return ""
    try:
        results = chroma_collection.query(query_texts=[user_message], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        return "\n\nRelevant nutrition info from our dataset:\n" + "\n".join(f"- {d}" for d in docs)
    except Exception as e:
        logger.warning(f"Chroma query failed: {e}")
        return ""

# ─────────────────────────────────────────────
# Vertex AI helper — call Llama via OpenAI-compatible endpoint
# ─────────────────────────────────────────────
def call_vertex_ai(system_prompt: str, history: list[dict], user_message: str) -> str:
    client = get_vertex_openai_client()

    # Build messages list: system prompt + conversation history + new message
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=VERTEX_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "chroma_connected": chroma_collection is not None,
        "vertex_model": VERTEX_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/auth/signup")
async def signup(req: SignupRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/signup",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": req.email, "password": req.password},
        )

    if resp.status_code not in (200, 201):
        detail = resp.json().get("error_description") or resp.json().get("msg") or "Signup failed."
        raise HTTPException(status_code=400, detail=detail)

    data         = resp.json()
    user_id      = data["user"]["id"]
    session      = data.get("session") or {}
    access_token = data.get("access_token") or session.get("access_token")

    admin_supabase.table("user_profiles").upsert({
        "user_id":             user_id,
        "health_conditions":   req.health_conditions,
        "dietary_preferences": req.dietary_preferences,
        "created_at":          datetime.now(timezone.utc).isoformat(),
    }).execute()

    logger.info(f"New signup: {user_id}")
    return {"user_id": user_id, "access_token": access_token, "message": "Account created."}


@app.post("/auth/login")
async def login(req: LoginRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": req.email, "password": req.password},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    data = resp.json()
    return {
        "user_id":       data["user"]["id"],
        "access_token":  data["access_token"],
        "refresh_token": data.get("refresh_token"),
    }


@app.post("/chat/session")
async def create_session(req: SessionRequest, user=Depends(get_current_user)):
    result = admin_supabase.table("chat_sessions").insert({
        "user_id":    user["id"],
        "title":      req.title,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()

    session = result.data[0]
    return {"session_id": session["id"], "title": session["title"]}


@app.post("/chat/message")
async def send_message(req: MessageRequest, user=Depends(get_current_user)):
    user_id = user["id"]

    # 2. Health profile
    profile_result = (
        admin_supabase.table("user_profiles")
        .select("health_conditions, dietary_preferences")
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    profile    = profile_result.data or {}
    conditions = profile.get("health_conditions") or []
    prefs      = profile.get("dietary_preferences") or []

    # 3. System prompt
    conditions_str = ", ".join(conditions) if conditions else "no specific health conditions"
    prefs_str      = ", ".join(prefs)      if prefs      else "no specific dietary restrictions"

    system_prompt = (
        f"You are TasteMatch, a helpful and friendly nutrition assistant. "
        f"This user has the following health conditions: {conditions_str}. "
        f"Their dietary preferences: {prefs_str}. "
        f"Always tailor your food and recipe advice to their specific health needs. "
        f"Be warm, encouraging, and practical. Keep responses to 2-3 paragraphs."
    )

    # 4. RAG context
    rag_context = get_rag_context(req.message)
    if rag_context:
        system_prompt += rag_context

    # 5. Conversation history (last 10 turns)
    history_result = (
        admin_supabase.table("chat_messages")
        .select("role, content")
        .eq("session_id", req.session_id)
        .order("created_at", desc=False)
        .limit(10)
        .execute()
    )
    history = history_result.data or []

    # 6. Call Vertex AI (Llama via OpenAI-compatible endpoint)
    try:
        ai_response = call_vertex_ai(system_prompt, history, req.message)
    except Exception as e:
        logger.error(f"Vertex AI error: {e}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")

    # 7. Save to Supabase
    now = datetime.now(timezone.utc).isoformat()
    admin_supabase.table("chat_messages").insert([
        {"session_id": req.session_id, "user_id": user_id, "role": "user",      "content": req.message,  "created_at": now},
        {"session_id": req.session_id, "user_id": user_id, "role": "assistant",  "content": ai_response,  "created_at": now},
    ]).execute()

    return {
        "response": ai_response,
        "rag_used": chroma_collection is not None and bool(rag_context),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
