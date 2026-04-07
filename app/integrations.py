"""
External integrations — Tandoor and Mealie recipe push.
"""

import re
import httpx


# ── Tandoor ────────────────────────────────────────────────────────────────────
def _build_tandoor_ingredients(recipe: dict) -> list:
    return [
        {
            "food":      {"name": ing["food"]},
            "unit":      {"name": ing["unit"]} if ing.get("unit") else None,
            "amount":    ing.get("amount", 0),
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
        "servings":     recipe.get("servings", 4),
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
