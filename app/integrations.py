"""
External integrations — Tandoor and Mealie recipe push.
"""

import re
import httpx


# ── Tandoor ────────────────────────────────────────────────────────────────────
def _to_float(val) -> float:
    """Coerce ingredient amounts (including unicode fractions) to float.

    Handles: plain numbers, unicode fractions (½ ⅔ ¼ etc.),
    compound values like "1½", "2⅔", and slash fractions like "1/2".
    """
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s:
        return 0.0

    # Unicode fraction character values
    UNICODE_FRACTIONS = {
        "½": 0.5,        # ½
        "⅓": 1/3,        # ⅓
        "⅔": 2/3,        # ⅔
        "¼": 0.25,       # ¼
        "¾": 0.75,       # ¾
        "⅛": 0.125,      # ⅛
        "⅜": 0.375,      # ⅜
        "⅝": 0.625,      # ⅝
        "⅞": 0.875,      # ⅞
        "⅕": 0.2,        # ⅕
        "⅖": 0.4,        # ⅖
        "⅗": 0.6,        # ⅗
        "⅘": 0.8,        # ⅘
        "⅙": 1/6,        # ⅙
        "⅚": 5/6,        # ⅚
    }

    # Check if the whole string is a unicode fraction
    if s in UNICODE_FRACTIONS:
        return UNICODE_FRACTIONS[s]

    # Check for compound: whole number + unicode fraction e.g. "1½", "2⅔"
    for frac_char, frac_val in UNICODE_FRACTIONS.items():
        if s.endswith(frac_char):
            whole_part = s[:-len(frac_char)].strip()
            if whole_part:
                try:
                    return float(whole_part) + frac_val
                except ValueError:
                    pass

    # Handle slash fractions e.g. "1/2", "3/4"
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0].strip()) / float(parts[1].strip())
            except (ValueError, ZeroDivisionError):
                pass

    # Plain number
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _build_tandoor_ingredients(recipe: dict) -> list:
    return [
        {
            "food":      {"name": ing["food"]},
            "unit":      {"name": ing["unit"]} if ing.get("unit") else None,
            "amount":    _to_float(ing.get("amount", 0)),
            "note":      ing.get("note", ""),
            "order":     0,
            "is_header": False,
        }
        for ing in recipe.get("ingredients", [])
    ]


def _build_tandoor_steps(recipe: dict, ingredients: list) -> list:
    steps = []
    if ingredients:
        steps.append({
            "name":        "Ingredients",
            "instruction": "",
            "ingredients": ingredients,
            "time":        0,
            "order":       0,
            "step_recipe": None,
        })
    for i, step in enumerate(recipe.get("steps", [])):
        steps.append({
            "name":        "",
            "instruction": step["text"],
            "ingredients": [],
            "time":        step.get("time", 0),
            "order":       i + 1,
            "step_recipe": None,
        })
    return steps


async def push_to_tandoor(recipe: dict, thumbnail: bytes | None,
                          tandoor_url: str, tandoor_token: str) -> dict:
    """Push recipe to Tandoor. Returns {success, recipe_id, url} or {success, error}."""
    if not tandoor_url or not tandoor_token:
        return {"success": False, "error": "Tandoor is not configured"}

    headers = {"Authorization": f"Bearer {tandoor_token}"}
    ingredients = _build_tandoor_ingredients(recipe)
    steps       = _build_tandoor_steps(recipe, ingredients)

    payload = {
        "name":         recipe["name"],
        "description":  recipe.get("description", ""),
        "servings":     recipe.get("servings") or 4,
        "working_time": recipe.get("prepTime", 0),
        "waiting_time": recipe.get("cookTime", 0),
        "keywords":     [{"name": k} for k in recipe.get("keywords", [])],
        "steps":        steps,
        "private":      False,
        "source_url":   "",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{tandoor_url}/api/recipe/",
            json=payload,
            headers={**headers, "Content-Type": "application/json"},
        )
        if resp.status_code not in (200, 201):
            return {"success": False, "error": resp.text}

        data = resp.json()
        rid  = data.get("id")

        if rid and thumbnail:
            try:
                files = {"image": ("thumbnail.jpg", thumbnail, "image/jpeg")}
                await client.put(
                    f"{tandoor_url}/api/recipe/{rid}/image/",
                    files=files,
                    headers=headers,
                )
            except Exception as e:
                print(f"[tandoor] Thumbnail upload failed (non-fatal): {e}")

    return {"success": True, "recipe_id": rid, "url": f"{tandoor_url}/view/recipe/{rid}"}


# ── Mealie ─────────────────────────────────────────────────────────────────────
def _mealie_ingredient_display(ing: dict) -> str:
    parts = []
    amount = ing.get("amount") or 0
    if amount > 0:
        parts.append(str(int(amount)) if amount == int(amount) else str(amount))
    if ing.get("unit"):
        parts.append(ing["unit"])
    if ing.get("food"):
        parts.append(ing["food"])
    if ing.get("note"):
        parts.append(f"({ing['note']})")
    return " ".join(parts)


async def push_to_mealie(recipe: dict, thumbnail: bytes | None,
                         mealie_url: str, mealie_token: str) -> dict:
    """Push recipe to Mealie. Returns {success, slug, url} or {success, error}."""
    if not mealie_url or not mealie_token:
        return {"success": False, "error": "Mealie is not configured"}

    headers = {"Authorization": f"Bearer {mealie_token}"}

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1 — create shell
        create_resp = await client.post(
            f"{mealie_url}/api/recipes",
            json={"name": recipe["name"]},
            headers={**headers, "Content-Type": "application/json"},
        )
        if create_resp.status_code not in (200, 201):
            return {"success": False, "error": create_resp.text}

        slug = create_resp.json()

        # Fetch group slug
        group_slug = None
        try:
            group_resp = await client.get(f"{mealie_url}/api/groups/self", headers=headers)
            if group_resp.status_code == 200:
                group_slug = group_resp.json().get("slug")
        except Exception:
            pass

        # Step 2 — PATCH full details
        prep_mins = recipe.get("prepTime", 0)
        cook_mins = recipe.get("cookTime", 0)

        def _ing_obj(ing):
            full = _mealie_ingredient_display(ing)
            return {"originalText": full, "note": full, "display": full}

        payload = {
            "name":        recipe["name"],
            "description": recipe.get("description", ""),
            "recipeYield": str(recipe.get("servings", 4)),
            "tags":        [],
            "recipeIngredient": [_ing_obj(ing) for ing in recipe.get("ingredients", [])],
            "recipeInstructions": [
                {"title": "", "text": step["text"], "ingredientReferences": []}
                for step in recipe.get("steps", [])
            ],
        }
        if prep_mins:
            payload["prepTime"] = f"PT{prep_mins}M"
        if cook_mins:
            payload["performTime"] = f"PT{cook_mins}M"
        if prep_mins or cook_mins:
            payload["totalTime"] = f"PT{prep_mins + cook_mins}M"

        patch_resp = await client.patch(
            f"{mealie_url}/api/recipes/{slug}",
            json=payload,
            headers={**headers, "Content-Type": "application/json"},
        )
        if patch_resp.status_code not in (200, 201):
            return {"success": False, "error": patch_resp.text}

        # Step 3 — thumbnail
        if thumbnail:
            try:
                files = {"image": ("thumbnail.jpg", thumbnail, "image/jpeg")}
                await client.put(
                    f"{mealie_url}/api/recipes/{slug}/image",
                    files=files,
                    headers=headers,
                )
            except Exception as e:
                print(f"[mealie] Thumbnail upload failed (non-fatal): {e}")

    if group_slug:
        recipe_url = f"{mealie_url}/g/{group_slug}/r/{slug}"
    else:
        recipe_url = f"{mealie_url}/recipe/{slug}"
    return {"success": True, "slug": slug, "url": recipe_url}
