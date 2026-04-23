# ReelMeals v4 — Text Recipe Import

import anthropic
import json
import re
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from auth import require_user

router = APIRouter()

TEXT_PARSE_SYSTEM_PROMPT = """You are a recipe extraction assistant. Your job is to parse raw recipe text — which may come from social media posts, blogs, or other informal sources — and extract it into a clean, structured JSON object.

Return ONLY valid JSON with no preamble, no markdown fences, no explanation.

The JSON schema is:
{
  "title": "string",
  "description": "string (brief, 1-2 sentences; derive from context or leave empty string)",
  "servings": integer or null,
  "prep_time": integer or null,
  "cook_time": integer or null,
  "total_time": integer or null,
  "source": "text_import",
  "ingredients": [
    {
      "quantity": "string or null",
      "unit": "string or null",
      "name": "string",
      "note": "string or null"
    }
  ],
  "steps": [
    {
      "order": integer,
      "text": "string"
    }
  ],
  "tags": ["string"]
}

Rules:
- Consolidate multi-section ingredient lists (e.g. "For the base:", "For the frosting:") by prepending the section name to the ingredient note field where useful for clarity.
- Break directions into discrete steps — one action per step where possible.
- Strip social media noise (emoji prompts, "Comment X to get Y", calls to action, hashtags).
- Infer servings/times only if clearly stated; otherwise use null.
- Unicode fractions (⅔, ½, ¼) are valid — preserve them as-is in quantity field.
"""


class TextImportRequest(BaseModel):
    raw_text: str


class TextImportResponse(BaseModel):
    recipe: dict
    warnings: list[str] = []


@router.post("/api/recipes/import-text", response_model=TextImportResponse)
async def import_recipe_from_text(
    payload: TextImportRequest,
    request: Request,
    current_user: str = Depends(require_user),
):
    import users as users_module
    settings = users_module.get_settings(current_user)
    api_key = settings.get("anthropic_api_key", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="Anthropic API key not configured. Add it in Settings.")

    raw = payload.raw_text.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="No text provided.")
    if len(raw) > 20_000:
        raise HTTPException(status_code=400, detail="Text too long (max 20,000 chars).")

    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=TEXT_PARSE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Parse this recipe:\n\n{raw}",
                }
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {str(e)}")

    raw_response = message.content[0].text.strip()

    raw_response = re.sub(r"^```(?:json)?\s*", "", raw_response)
    raw_response = re.sub(r"\s*```$", "", raw_response)

    try:
        recipe = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse Claude response as JSON: {str(e)}",
        )

    if "ingredients" in recipe:
        normalized = []
        for ing in recipe["ingredients"]:
            normalized.append({
                "amount": ing.get("quantity") or ing.get("amount") or 0,
                "unit":   ing.get("unit") or "",
                "food":   ing.get("name") or ing.get("food") or "",
                "note":   ing.get("note") or "",
            })
        recipe["ingredients"] = normalized

    if "steps" in recipe:
        normalized_steps = []
        for step in recipe["steps"]:
            normalized_steps.append({
                "text": step.get("text") or step.get("instruction") or "",
                "time": step.get("time") or 0,
            })
        recipe["steps"] = normalized_steps

    if not recipe.get("name") and recipe.get("title"):
        recipe["name"] = recipe["title"]

    if not recipe.get("keywords") and recipe.get("tags"):
        recipe["keywords"] = recipe["tags"]

    if not recipe.get("prepTime") and recipe.get("prep_time"):
        recipe["prepTime"] = recipe["prep_time"]
    if not recipe.get("cookTime") and recipe.get("cook_time"):
        recipe["cookTime"] = recipe["cook_time"]

    warnings = []
    if not recipe.get("name") and not recipe.get("title"):
        warnings.append("Could not extract a recipe title — please set one before saving.")
    if not recipe.get("ingredients"):
        warnings.append("No ingredients found — check the parsed result.")
    if not recipe.get("steps"):
        warnings.append("No steps found — check the parsed result.")

    return TextImportResponse(recipe=recipe, warnings=warnings)
