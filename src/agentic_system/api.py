import requests
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from agentic_system.config.settings import get_settings
from agentic_system.orchestrator.graph import Orchestrator

app = FastAPI(title="Agentic System API", docs_url=None, redoc_url=None)
router = APIRouter(prefix="/api")
orchestrator = Orchestrator()
settings = get_settings()


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/api/docs")


@router.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={"theme": "dark"},
    )


class InvokeRequest(BaseModel):
    prompt: str


class InvokeResponse(BaseModel):
    response: str


class EnhanceSkillRequest(BaseModel):
    title: str
    description: str


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/invoke", response_model=InvokeResponse)
async def invoke_agent(request: InvokeRequest):
    try:
        result = orchestrator.invoke(request.prompt)
        return InvokeResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enhance-skill", response_model=InvokeResponse)
async def enhance_skill(request: EnhanceSkillRequest):
    try:
        # Explicitly target the skill_enhancer agent with both title and description
        prompt = f"Skill Title: {request.title}\nDescription: {request.description}"
        result = orchestrator.invoke(prompt, agent_id="skill_enhancer")
        return InvokeResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get-models")
def get_models():
    """Fetches available models from the configured provider."""
    if settings.llm_provider == "gemini":
        if not settings.google_api_key:
            raise HTTPException(status_code=400, detail="GOOGLE_API_KEY is not set")
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={settings.google_api_key}"
        response = requests.get(url)
    elif settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set")
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported provider: {settings.llm_provider}"
        )

    return response.json()


app.include_router(router)
