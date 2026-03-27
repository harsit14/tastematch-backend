"""
TasteMatch Pilot — FastAPI Backend (Phase 4, v2)
Full implementation with expanded schema support.

Auth:
  POST /auth/signup
  POST /auth/login

User:
  GET  /profile
  PUT  /profile

Chat:
  POST /chat/session
  GET  /chat/sessions
  GET  /chat/sessions/{session_id}/messages
  POST /chat/message

Glucose:
  POST /glucose
  GET  /glucose

Meals:
  POST /meals
  GET  /meals

Body Metrics:
  POST /body-metrics
  GET  /body-metrics

Fridge:
  GET    /fridge
  POST   /fridge
  DELETE /fridge/{item_id}

Foods:
  GET /foods/search?q=...

Recipes:
  POST   /recipes/save
  GET    /recipes
  DELETE /recipes/{recipe_id}

System:
  GET /health
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import openai
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
# Config
# ─────────────────────────────────────────────
SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_ANON_KEY    = os.environ["SUPABASE_ANON_KEY"]

VERTEX_PROJECT  = os.environ.get("VERTEX_PROJECT", "tastematch-pilot")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL    = os.environ.get("VERTEX_MODEL", "meta/llama-3.3-70b-instruct-maas")

# USDA FoodData Central API key — get free at https://fdc.nal.usda.gov/api-guide.html
USDA_API_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")

PORT = int(os.environ.get("PORT", "8000"))

# ─────────────────────────────────────────────
# Supabase admin client
# ─────────────────────────────────────────────
admin_supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─────────────────────────────────────────────
# Vertex AI — OpenAI-compatible client for Llama MaaS
# ─────────────────────────────────────────────
def get_vertex_openai_client() -> openai.OpenAI:
    creds_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
    creds_file     = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    if creds_json_str:
        creds_dict  = json.loads(creds_json_str)
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    elif creds_file and os.path.exists(creds_file):
        credentials = service_account.Credentials.from_service_account_file(
            creds_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    else:
        credentials, _ = google_auth_default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    credentials.refresh(GoogleAuthRequest())

    return openai.OpenAI(
        base_url=(
            f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1beta1"
            f"/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/endpoints/openapi"
        ),
        api_key=credentials.token,
    )

# ─────────────────────────────────────────────
# Vertex AI text embeddings
# ─────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """Generate a 768-dim embedding using Vertex AI text-embedding-004."""
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        creds_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
        if creds_json_str:
            from google.oauth2 import service_account
            creds_dict  = json.loads(creds_json_str)
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            vertexai.init(
                project=VERTEX_PROJECT,
                location=VERTEX_LOCATION,
                credentials=credentials
            )
        else:
            vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        result = model.get_embeddings([text])
        return result[0].values
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return []

      
# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
security = HTTPBearer()

app = FastAPI(title="TasteMatch API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    date_of_birth: Optional[str] = None       # ISO date string e.g. "1990-05-20"
    diabetes_type: Optional[str] = None        # e.g. "type2", "type1", "prediabetes"
    dietary_preferences: list[str] = []
    allergies: list[str] = []

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ProfileUpdateRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    date_of_birth: Optional[str] = None
    diabetes_type: Optional[str] = None
    dietary_preferences: Optional[list[str]] = None
    allergies: Optional[list[str]] = None

class SessionRequest(BaseModel):
    title: Optional[str] = "New conversation"

class MessageRequest(BaseModel):
    session_id: str
    message: str

class GlucoseRequest(BaseModel):
    reading_mmol: float                        # blood sugar in mmol/L
    reading_context: Optional[str] = None      # e.g. "before_meal", "after_meal", "fasting"
    notes: Optional[str] = None
    recorded_at: Optional[str] = None          # ISO datetime, defaults to now

class MealLogRequest(BaseModel):
    meal_name: str
    carbs_grams: Optional[float] = None
    calories: Optional[float] = None
    meal_type: Optional[str] = None            # e.g. "breakfast", "lunch", "dinner", "snack"
    notes: Optional[str] = None
    eaten_at: Optional[str] = None             # ISO datetime, defaults to now

class BodyMetricsRequest(BaseModel):
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    bmi: Optional[float] = None
    recorded_at: Optional[str] = None

class FridgeItemRequest(BaseModel):
    food_id: Optional[str] = None             # uuid from foods table (if known)
    food_name: str                             # human-readable name for display
    quantity: Optional[float] = None
    unit: Optional[str] = None                # e.g. "g", "ml", "pieces"
    expiry_date: Optional[str] = None         # ISO date string

class SaveRecipeRequest(BaseModel):
    title: str
    ingredients: list[dict]                   # jsonb — list of {name, quantity, unit}
    instructions: Optional[str] = None
    carbs_per_serving: Optional[float] = None
    calories_per_serving: Optional[float] = None
    servings: Optional[int] = None
    tags: list[str] = []
    chroma_doc_id: Optional[str] = None       # link back to RAG dataset doc

# ─────────────────────────────────────────────
# Auth helper
# ─────────────────────────────────────────────
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return resp.json()

# ─────────────────────────────────────────────
# RAG helper
# ─────────────────────────────────────────────
def get_rag_context(user_message: str, n_results: int = 3) -> str:
    """Retrieve relevant nutrition docs from Supabase pgvector."""
    try:
        embedding = get_embedding(user_message)
        if not embedding:
            return ""

        result = admin_supabase.rpc("match_nutrition_documents", {
            "query_embedding": embedding,
            "match_count": n_results,
        }).execute()

        docs = result.data or []
        if not docs:
            return ""

        return "\n\nRelevant nutrition info from our knowledge base:\n" + \
               "\n".join(f"- {d['content']}" for d in docs)
    except Exception as e:
        logger.warning(f"pgvector RAG query failed: {e}")
        return ""

# ─────────────────────────────────────────────
# Vertex AI helper
# ─────────────────────────────────────────────
def call_vertex_ai(system_prompt: str, history: list[dict], user_message: str) -> str:
    client = get_vertex_openai_client()

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
# Food API helpers
# ─────────────────────────────────────────────
async def search_usda(query: str, limit: int = 5) -> list[dict]:
    """Search USDA FoodData Central. Free API key at fdc.nal.usda.gov"""
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params={
                "query": query,
                "pageSize": limit,
                "api_key": USDA_API_KEY,
            }, timeout=10)

        if resp.status_code != 200:
            return []

        foods = resp.json().get("foods", [])
        results = []
        for f in foods:
            nutrients = {n["nutrientName"]: n.get("value") for n in f.get("foodNutrients", [])}
            results.append({
                "external_id": str(f.get("fdcId")),
                "source": "usda",
                "name": f.get("description", ""),
                "brand": f.get("brandOwner"),
                "calories_per_100g": nutrients.get("Energy"),
                "protein_per_100g": nutrients.get("Protein"),
                "carbs_per_100g": nutrients.get("Carbohydrate, by difference"),
                "fat_per_100g": nutrients.get("Total lipid (fat)"),
                "fiber_per_100g": nutrients.get("Fiber, total dietary"),
                "serving_size_g": f.get("servingSize"),
            })
        return results
    except Exception as e:
        logger.warning(f"USDA search failed: {e}")
        return []


async def search_open_food_facts(query: str, limit: int = 5) -> list[dict]:
    """Search Open Food Facts — no API key needed, best for packaged/branded products."""
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params={
                "search_terms": query,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": limit,
            }, timeout=10)

        if resp.status_code != 200:
            return []

        products = resp.json().get("products", [])
        results = []
        for p in products:
            nutriments = p.get("nutriments", {})
            name = p.get("product_name") or p.get("product_name_en")
            if not name:
                continue
            results.append({
                "external_id": p.get("code"),
                "source": "open_food_facts",
                "name": name,
                "brand": p.get("brands"),
                "calories_per_100g": nutriments.get("energy-kcal_100g"),
                "protein_per_100g": nutriments.get("proteins_100g"),
                "carbs_per_100g": nutriments.get("carbohydrates_100g"),
                "fat_per_100g": nutriments.get("fat_100g"),
                "fiber_per_100g": nutriments.get("fiber_100g"),
                "serving_size_g": nutriments.get("serving_size"),
            })
        return results
    except Exception as e:
        logger.warning(f"Open Food Facts search failed: {e}")
        return []


async def save_food_to_db(food: dict) -> str:
    """Save a food item to the foods table if not already there. Returns the food id."""
    existing = (
        admin_supabase.table("foods")
        .select("id")
        .eq("external_id", food["external_id"])
        .eq("source", food["source"])
        .execute()
    )
    if existing.data:
        return existing.data[0]["id"]

    result = admin_supabase.table("foods").insert({
        **food,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return result.data[0]["id"]

# ─────────────────────────────────────────────
# Context builder for chat
# ─────────────────────────────────────────────
def build_user_context(user_id: str) -> str:
    """
    Pulls user profile, recent glucose readings, recent meals, and fridge
    contents to inject as context into the system prompt.
    """
    context_parts = []

    # Profile
    try:
        profile = (
            admin_supabase.table("user_profiles")
            .select("first_name, diabetes_type, dietary_preferences, allergies")
            .eq("id", user_id)
            .single()
            .execute()
        ).data or {}

        name          = profile.get("first_name") or "the user"
        diabetes      = profile.get("diabetes_type")
        prefs         = profile.get("dietary_preferences") or []
        allergies     = profile.get("allergies") or []

        context_parts.append(
            f"User: {name}. "
            + (f"Diabetes type: {diabetes}. " if diabetes else "")
            + (f"Dietary preferences: {', '.join(prefs)}. " if prefs else "")
            + (f"Allergies: {', '.join(allergies)}. " if allergies else "")
        )
    except Exception as e:
        logger.warning(f"Could not fetch profile for context: {e}")

    # Recent glucose readings (last 3)
    try:
        glucose = (
            admin_supabase.table("glucose_readings")
            .select("reading_mmol, reading_context, recorded_at")
            .eq("user_id", user_id)
            .order("recorded_at", desc=True)
            .limit(3)
            .execute()
        ).data or []

        if glucose:
            readings_str = ", ".join(
                f"{r['reading_mmol']} mmol/L ({r.get('reading_context', 'unspecified')})"
                for r in glucose
            )
            context_parts.append(f"Recent glucose readings: {readings_str}.")
    except Exception as e:
        logger.warning(f"Could not fetch glucose for context: {e}")

    # Recent meals (last 3)
    try:
        meals = (
            admin_supabase.table("meal_logs")
            .select("meal_name, carbs_grams, calories, meal_type, eaten_at")
            .eq("user_id", user_id)
            .order("eaten_at", desc=True)
            .limit(3)
            .execute()
        ).data or []

        if meals:
            meals_str = ", ".join(
                f"{m['meal_name']} ({m.get('meal_type', '')})"
                for m in meals
            )
            context_parts.append(f"Recent meals logged: {meals_str}.")
    except Exception as e:
        logger.warning(f"Could not fetch meals for context: {e}")

    # Fridge contents
    try:
        fridge = (
            admin_supabase.table("fridge_items")
            .select("food_name, quantity, unit, expiry_date")
            .eq("user_id", user_id)
            .execute()
        ).data or []

        if fridge:
            fridge_str = ", ".join(
                f"{item['food_name']}"
                + (f" ({item['quantity']} {item.get('unit', '')})" if item.get("quantity") else "")
                for item in fridge
            )
            context_parts.append(f"Items currently in their fridge: {fridge_str}.")
    except Exception as e:
        logger.warning(f"Could not fetch fridge for context: {e}")

    return "\n".join(context_parts)

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ── System ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rag": "pgvector",
        "vertex_model": VERTEX_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ── Auth ──────────────────────────────────────────────────────────────────────

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
    now = datetime.now(timezone.utc).isoformat()
    calculated_age = None
    if req.date_of_birth:
      from datetime import date
      dob = date.fromisoformat(req.date_of_birth)
      today = date.today()
      calculated_age = today.year - dob.year - (
        (today.month, today.day) < (dob.month, dob.day)
      )
    admin_supabase.table("user_profiles").upsert({
        "id":                  user_id,
        "first_name":          req.first_name,
        "last_name":           req.last_name,
        "age":                 calculated_age,
        "date_of_birth":       req.date_of_birth,
        "diabetes_type":       req.diabetes_type,
        "dietary_preferences": req.dietary_preferences,
        "allergies":           req.allergies,
        "created_at":          now,
        "updated_at":          now,
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

# ── Profile ───────────────────────────────────────────────────────────────────

@app.get("/profile")
async def get_profile(user=Depends(get_current_user)):
    result = (
        admin_supabase.table("user_profiles")
        .select("*")
        .eq("id", user["id"])
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return result.data


@app.put("/profile")
async def update_profile(req: ProfileUpdateRequest, user=Depends(get_current_user)):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    result = (
        admin_supabase.table("user_profiles")
        .update(updates)
        .eq("id", user["id"])
        .execute()
    )
    return result.data[0]

# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat/session")
async def create_session(req: SessionRequest, user=Depends(get_current_user)):
    now    = datetime.now(timezone.utc).isoformat()
    result = admin_supabase.table("chat_sessions").insert({
        "user_id":    user["id"],
        "title":      req.title,
        "created_at": now,
        "updated_at": now,
    }).execute()

    session = result.data[0]
    return {"session_id": session["id"], "title": session["title"]}


@app.get("/chat/sessions")
async def get_sessions(user=Depends(get_current_user)):
    result = (
        admin_supabase.table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user["id"])
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data or []


@app.get("/chat/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user=Depends(get_current_user)):
    # Verify session belongs to this user
    session = (
        admin_supabase.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user["id"])
        .execute()
    )
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found.")

    result = (
        admin_supabase.table("chat_messages")
        .select("id, session_id, role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data or []


@app.post("/chat/message")
async def send_message(req: MessageRequest, user=Depends(get_current_user)):
    user_id = user["id"]

    # Build rich user context from all data sources
    user_context = build_user_context(user_id)

    system_prompt = (
        "You are TasteMatch, a helpful and friendly nutrition assistant specialising in "
        "diabetes-aware meal planning and healthy eating. "
        "Use the user's health data below to personalise every response. "
        "When the user has items in their fridge, suggest recipes using those ingredients. "
        "When glucose readings are high, recommend low-glycaemic options. "
        "When suggesting recipes, assess their likely glycemic impact based on "
        "ingredients and cooking method even when explicit GI values are not listed. "
        "Flag high-GI ingredients and suggest lower-GI swaps where appropriate. "
        "Be warm, practical, and concise — 2 to 3 paragraphs max.\n\n"
        f"{user_context}"
    )

    # RAG context from Chroma
    rag_context = get_rag_context(req.message)
    if rag_context:
        system_prompt += rag_context

    # Conversation history
    history = (
        admin_supabase.table("chat_messages")
        .select("role, content")
        .eq("session_id", req.session_id)
        .order("created_at", desc=False)
        .limit(10)
        .execute()
    ).data or []

    # Call Vertex AI
    try:
        ai_response = call_vertex_ai(system_prompt, history, req.message)
    except Exception as e:
        logger.error(f"Vertex AI error: {e}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")

    # Save messages
    now = datetime.now(timezone.utc).isoformat()
    admin_supabase.table("chat_messages").insert([
        {"session_id": req.session_id, "user_id": user_id, "role": "user",      "content": req.message,  "created_at": now},
        {"session_id": req.session_id, "user_id": user_id, "role": "assistant", "content": ai_response,  "created_at": now},
    ]).execute()

    # Update session updated_at
    admin_supabase.table("chat_sessions").update(
        {"updated_at": now}
    ).eq("id", req.session_id).execute()

    return {
        "response": ai_response,
        "rag_used": bool(rag_context),
    }

# ── Glucose ───────────────────────────────────────────────────────────────────

@app.post("/glucose")
async def log_glucose(req: GlucoseRequest, user=Depends(get_current_user)):
    now = datetime.now(timezone.utc).isoformat()
    result = admin_supabase.table("glucose_readings").insert({
        "user_id":         user["id"],
        "reading_mmol":    req.reading_mmol,
        "reading_context": req.reading_context,
        "notes":           req.notes,
        "recorded_at":     req.recorded_at or now,
        "created_at":      now,
    }).execute()
    return result.data[0]


@app.get("/glucose")
async def get_glucose(
    limit: int = Query(20, ge=1, le=100),
    user=Depends(get_current_user)
):
    result = (
        admin_supabase.table("glucose_readings")
        .select("*")
        .eq("user_id", user["id"])
        .order("recorded_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []

# ── Meals ─────────────────────────────────────────────────────────────────────

@app.post("/meals")
async def log_meal(req: MealLogRequest, user=Depends(get_current_user)):
    now = datetime.now(timezone.utc).isoformat()
    result = admin_supabase.table("meal_logs").insert({
        "user_id":     user["id"],
        "meal_name":   req.meal_name,
        "carbs_grams": req.carbs_grams,
        "calories":    req.calories,
        "meal_type":   req.meal_type,
        "notes":       req.notes,
        "eaten_at":    req.eaten_at or now,
        "created_at":  now,
    }).execute()
    return result.data[0]


@app.get("/meals")
async def get_meals(
    limit: int = Query(20, ge=1, le=100),
    user=Depends(get_current_user)
):
    result = (
        admin_supabase.table("meal_logs")
        .select("*")
        .eq("user_id", user["id"])
        .order("eaten_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []

# ── Body Metrics ──────────────────────────────────────────────────────────────

@app.post("/body-metrics")
async def log_body_metrics(req: BodyMetricsRequest, user=Depends(get_current_user)):
    now = datetime.now(timezone.utc).isoformat()


    result = admin_supabase.table("body_metrics").insert({
        "user_id":     user["id"],
        "weight_kg":   req.weight_kg,
        "height_cm":   req.height_cm,
        "recorded_at": req.recorded_at or now,
    }).execute()
    return result.data[0]


@app.get("/body-metrics")
async def get_body_metrics(
    limit: int = Query(10, ge=1, le=50),
    user=Depends(get_current_user)
):
    result = (
        admin_supabase.table("body_metrics")
        .select("*")
        .eq("user_id", user["id"])
        .order("recorded_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []

# ── Fridge ────────────────────────────────────────────────────────────────────

@app.get("/fridge")
async def get_fridge(user=Depends(get_current_user)):
    result = (
        admin_supabase.table("fridge_items")
        .select("*, foods(name, calories_per_100g, carbs_per_100g, protein_per_100g)")
        .eq("user_id", user["id"])
        .execute()
    )
    return result.data or []


@app.post("/fridge")
async def add_fridge_item(req: FridgeItemRequest, user=Depends(get_current_user)):
    now = datetime.now(timezone.utc).isoformat()
    result = admin_supabase.table("fridge_items").insert({
        "user_id":     user["id"],
        "food_id":     req.food_id,
        "food_name":   req.food_name,
        "quantity":    req.quantity,
        "unit":        req.unit,
        "expiry_date": req.expiry_date,
        "added_at":    now,
    }).execute()
    return result.data[0]


@app.delete("/fridge/{item_id}")
async def remove_fridge_item(item_id: str, user=Depends(get_current_user)):
    # Verify the item belongs to this user before deleting
    existing = (
        admin_supabase.table("fridge_items")
        .select("id")
        .eq("id", item_id)
        .eq("user_id", user["id"])
        .execute()
    )
    if not existing.data:
        raise HTTPException(status_code=404, detail="Item not found.")

    admin_supabase.table("fridge_items").delete().eq("id", item_id).execute()
    return {"message": "Item removed from fridge."}

# ── Foods Search ──────────────────────────────────────────────────────────────

@app.get("/foods/search")
async def search_foods(
    q: str = Query(..., min_length=2, description="Food name to search"),
    limit: int = Query(5, ge=1, le=20),
    user=Depends(get_current_user)
):
    """
    Searches foods in this order:
    1. Local Supabase foods table (cached results)
    2. USDA FoodData Central (whole foods, ingredients)
    3. Open Food Facts (packaged/branded products)
    Results from external APIs are saved to the local foods table for future use.
    """

    # 1. Check local DB first
    local = (
        admin_supabase.table("foods")
        .select("*")
        .ilike("name", f"%{q}%")
        .limit(limit)
        .execute()
    ).data or []

    if len(local) >= limit:
        return {"source": "local", "results": local}

    # 2. Search USDA
    usda_results = await search_usda(q, limit)

    # 3. Search Open Food Facts
    off_results = await search_open_food_facts(q, limit)

    # Combine and deduplicate by name
    combined = usda_results + off_results
    seen_names = {item["name"].lower() for item in local}
    new_results = []

    for food in combined:
        if food["name"].lower() not in seen_names:
            seen_names.add(food["name"].lower())
            # Save to local DB for future use
            try:
                food_id = await save_food_to_db(food)
                food["id"] = food_id
            except Exception as e:
                logger.warning(f"Could not save food to DB: {e}")
            new_results.append(food)

    all_results = local + new_results
    return {"source": "api", "results": all_results[:limit]}

# ── Saved Recipes ─────────────────────────────────────────────────────────────

@app.post("/recipes/save")
async def save_recipe(req: SaveRecipeRequest, user=Depends(get_current_user)):
    now = datetime.now(timezone.utc).isoformat()
    result = admin_supabase.table("saved_recipes").insert({
        "user_id":              user["id"],
        "title":                req.title,
        "ingredients":          req.ingredients,
        "instructions":         req.instructions,
        "carbs_per_serving":    req.carbs_per_serving,
        "calories_per_serving": req.calories_per_serving,
        "servings":             req.servings,
        "tags":                 req.tags,
        "chroma_doc_id":        req.chroma_doc_id,
        "created_at":           now,
    }).execute()
    return result.data[0]


@app.get("/recipes")
async def get_saved_recipes(user=Depends(get_current_user)):
    result = (
        admin_supabase.table("saved_recipes")
        .select("*")
        .eq("user_id", user["id"])
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


@app.delete("/recipes/{recipe_id}")
async def delete_recipe(recipe_id: str, user=Depends(get_current_user)):
    existing = (
        admin_supabase.table("saved_recipes")
        .select("id")
        .eq("id", recipe_id)
        .eq("user_id", user["id"])
        .execute()
    )
    if not existing.data:
        raise HTTPException(status_code=404, detail="Recipe not found.")

    admin_supabase.table("saved_recipes").delete().eq("id", recipe_id).execute()
    return {"message": "Recipe deleted."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
