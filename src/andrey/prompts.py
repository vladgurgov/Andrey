"""Prompt templates and game profile loading."""

from pathlib import Path

import yaml

from andrey.models import ScreenInfo


def load_game_profile(profile_name: str) -> dict:
    """Load a game profile YAML file from game_profiles/ directory."""
    # Check local game_profiles/ first
    profile_path = Path("game_profiles") / f"{profile_name}.yaml"
    if not profile_path.exists():
        # Fallback to package-relative path
        profile_path = (
            Path(__file__).parent.parent.parent / "game_profiles" / f"{profile_name}.yaml"
        )

    if not profile_path.exists():
        raise FileNotFoundError(
            f"Game profile '{profile_name}' not found. "
            f"Looked in: game_profiles/{profile_name}.yaml"
        )

    with open(profile_path) as f:
        return yaml.safe_load(f)


def build_system_prompt(
    screen_info: ScreenInfo,
    game_profile: dict,
    omniparser_enabled: bool = True,
) -> str:
    """Build the system prompt for multi-turn tool-use conversation."""
    game_context = game_profile.get("system_context", "You are playing a mobile game.")
    game_rules = game_profile.get("rules", "")
    game_tips = game_profile.get("tips", "")

    rules_section = f"\n## Game Rules\n{game_rules}" if game_rules else ""
    tips_section = f"\n## Strategy Tips\n{game_tips}" if game_tips else ""

    w = screen_info.width
    h = screen_info.height

    if omniparser_enabled:
        perception_section = """## UI Element Detection
Each screenshot shows numbered bounding boxes around detected UI elements. You also receive a text list of elements with their IDs, types, descriptions, and whether they are interactive.

**ALWAYS use `tap_element(element_id=N)` to interact with elements.** This is the most reliable way to tap — it uses the exact center of the detected element's bounding box. To find the right element: look at the screenshot, find the item you want to tap, and read the number label drawn on or near it.

**IMPORTANT:** The element text descriptions are auto-generated and often wrong (e.g., a playing card may be described as "fire hydrant" or "simple symbol"). Ignore the descriptions — instead, match elements by their VISUAL POSITION in the screenshot. The numbered bounding boxes are drawn directly on the image, so you can see which number corresponds to which visual element.

Use `tap(x, y)` ONLY as a last resort when no element bounding box covers your target."""
    else:
        perception_section = f"""## Coordinate Estimation
There is no element detection — you must estimate pixel coordinates visually.
- Elements near the bottom of the screen have LARGE y values (close to {h})
- To estimate: visually locate the element, calculate its position as a fraction of {w}x{h}
- Example: an element 80% down the screen has y = {int(h * 0.8)}"""

    return f"""You are an AI agent playing a game on an Android phone. You analyze screenshots and take actions using the provided tools.

## Screen Information
- Screen resolution: {w}x{h} pixels
- Coordinate system: (0,0) is top-left, ({w},{h}) is bottom-right
- X increases left to right, Y increases top to bottom

{perception_section}

## Game Context
{game_context}
{rules_section}
{tips_section}

## How to Act
- You will receive screenshots of the current game state.
- Use the provided tools to interact with the game.
- You can take MULTIPLE actions per screenshot. After each action you will see the updated screen with fresh element detection.
- For multi-step actions (like bidding: select a number then tap PLAY), do each step as a separate tool call — you will see the result after each one.
- If the screen did NOT change after your action, your target was probably wrong. Try a different element or different coordinates.
- If it is NOT your turn (other players are acting), use the wait tool.
- If you see a loading screen or animation, use the wait tool.

## Important
- When the game ends (final score screen, game over), use the game_over tool.
- Dismiss ads by tapping X/close buttons or elements labeled as close/dismiss. Do NOT tap ad content.
- If a popup or dialog appears, handle it before continuing gameplay.
- Be precise — prefer tap_element over tap when the element is detected."""
