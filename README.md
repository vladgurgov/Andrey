# Andrey

AI agent that plays Android games autonomously — connects to an Android phone via ADB, detects UI elements with [OmniParser](https://github.com/microsoft/OmniParser), reasons about game state with Claude's vision API, and executes actions on the device in a multi-turn conversation loop.

<p align="center">
  <a href="https://www.youtube.com/shorts/9wH22hlTlms">
    <img src="https://img.youtube.com/vi/9wH22hlTlms/maxresdefault.jpg" width="300" alt="Demo Video">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/bidding_annotated.jpg" width="180" alt="OmniParser annotated screenshot">
</p>
<p align="center">
  <em>Left: early prototype in action (<a href="https://www.youtube.com/shorts/9wH22hlTlms">watch on YouTube</a>) · Right: what the agent sees (OmniParser bounding boxes)</em>
</p>

## How It Works

```
                            ANDREY AGENT LOOP
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   ┌─────────┐     ┌────────────┐     ┌───────────────┐  │
    │   │ Android │────>│ OmniParser │────>│    Claude      │  │
    │   │ Device  │ ADB │ V2 (local) │     │ Vision + Tools │  │
    │   │         │     │            │     │                │  │
    │   │  ┌───┐  │     │ YOLOv8     │     │ "I see bid 3  │  │
    │   │  │   │  │     │ Florence-2 │     │  at element 36,│  │
    │   │  │ # │  │     │ OCR        │     │  tapping it"  │  │
    │   │  └───┘  │     │            │     │                │  │
    │   └────▲────┘     └────────────┘     └───────┬───────┘  │
    │        │          screenshot with                │       │
    │        │          numbered boxes           tool call     │
    │        │               [5] [14]       tap_element(36)    │
    │        │               [36] [42]                │        │
    │        │                                        │        │
    │        └────────────────────────────────────────┘        │
    │                     execute on device                    │
    │                     wait for animation                   │
    │                     capture new screenshot               │
    └──────────────────────────────────────────────────────────┘
```

1. **Capture** — screenshot the phone via ADB, wait for animations to settle
2. **Detect** — OmniParser V2 (YOLOv8 + Florence-2 + OCR) draws numbered bounding boxes on every UI element
3. **Reason** — Claude sees the annotated image, reads box numbers visually, picks a tool: `tap_element(id=36)` or `tap(x, y)` as fallback
4. **Execute** — tap/swipe/wait on the device, capture result, feed back to Claude

The conversation is **multi-turn** — Claude remembers previous steps. A sliding window keeps the last 8 screenshots in context to manage token costs.

## Quick Start

```bash
# Install
pip install -e .
export ANTHROPIC_API_KEY="sk-ant-..."

# Play
andrey play --profile spades --steps 20 --save-annotated
```

**Requirements:** Python 3.10+, Android device with USB debugging, ADB running, [Anthropic API key](https://console.anthropic.com/)

## Installation

```bash
pip install -e .
```

### OmniParser Setup (recommended)

OmniParser runs locally on Apple Silicon (MPS), CUDA, or CPU. Adds ~3.5s per frame but gives exact element bounding boxes instead of guessed coordinates.

```bash
git clone https://github.com/microsoft/OmniParser.git ~/OmniParser
huggingface-cli download microsoft/OmniParser-v2.0 --local-dir ~/OmniParser/weights
pip install -e '.[omniparser]'
```

Point your `config.yaml` at it:
```yaml
omniparser:
  enabled: true
  omniparser_path: "/path/to/OmniParser"
  weights_path: "/path/to/OmniParser/weights"
  device: "mps"  # or "cuda" or "cpu"
```

Without OmniParser, the agent falls back to raw screenshots with coordinate-based tapping.

## Usage

```bash
andrey play                                        # default profile, 100 steps
andrey play --profile spades -v                    # verbose logging
andrey play --profile spades --steps 20 --delay 2  # 20 steps, 2s between actions
andrey play --no-omniparser                        # disable element detection
andrey play --save-annotated                       # save OmniParser debug images
andrey play --context "Bid conservatively"         # extra LLM context

andrey screenshot --save screen.png                # single screenshot
andrey devices                                     # list connected devices
andrey tap 540 1200                                # manual tap
```

### CLI Options

| Option | Description |
|---|---|
| `--profile, -p` | Game profile name (from `game_profiles/`) |
| `--device, -d` | ADB device serial |
| `--steps, -n` | Number of actions to execute (default 100) |
| `--delay` | Seconds between steps |
| `--context` | Extra context for the LLM |
| `--max-images` | Screenshots kept in context (default 8) |
| `--omniparser-path` | Path to OmniParser repo |
| `--no-omniparser` | Disable OmniParser |
| `--save-annotated` | Save annotated screenshots alongside raw |

## Game Profiles

YAML files in `game_profiles/` give the agent game-specific rules, strategy tips, and context.

```yaml
name: "Spades"
app_package: "com.youxi.spades"

system_context: |
  You are playing the card game Spades on an Android phone.

rules: |
  - Spades is always trump
  - You must follow suit if possible
  ...

tips: |
  - During bidding, count your high cards and spades
  - BIDDING UI: first tap the number element, then tap PLAY
  ...
```

A `default.yaml` profile is included for generic games.

## Configuration

Copy `config.example.yaml` to `config.yaml`:

```yaml
anthropic:
  api_key: ""          # or set ANTHROPIC_API_KEY env var
  model: "claude-sonnet-4-20250514"
  max_tokens: 1024

loop:
  delay_seconds: 1.5
  max_steps: 100
  error_threshold: 5

conversation:
  max_images: 8        # sliding window size

device:
  serial: null         # null = auto-detect
  adb_host: "127.0.0.1"
  adb_port: 5037

omniparser:
  enabled: true
  omniparser_path: ""  # auto-detect ~/OmniParser if empty
  device: "mps"        # mps, cuda, or cpu

game_profile: "default"
save_screenshots: true
save_annotated: false
screenshot_dir: "./screenshots"
```

## Session Logging

Every session writes `session.log` with full debug output — Claude's reasoning, tool calls, element lists, and results:

```
screenshots/
  session.log              # full debug log
  step_0000.jpg            # raw screenshot
  step_0000_annotated.jpg  # OmniParser overlay (--save-annotated)
  step_0001.jpg
  ...
```

## Architecture

```
src/andrey/
  agent.py       — observation-action loop + tool dispatch
  vision.py      — ConversationClient (multi-turn) + VisionClient (single-turn)
  omniparser.py  — OmniParser V2 wrapper, graceful fallback
  models.py      — tool definitions, ToolCall, ApiResponse, GameAction
  prompts.py     — system prompt builder + game profile loader
  device.py      — ADB device manager (screenshots, taps, swipes, keys)
  config.py      — Pydantic config models + YAML loading
  cli.py         — Click CLI entry point
  logger.py      — logging setup with file output
```

### Tools

| Tool | Description |
|---|---|
| `tap_element(element_id)` | Tap detected UI element by OmniParser ID *(preferred)* |
| `tap(x, y)` | Tap at pixel coordinates *(fallback)* |
| `swipe(x1, y1, x2, y2)` | Swipe gesture |
| `long_press(x, y)` | Long press |
| `press_key(BACK\|HOME\|ENTER)` | Android system key |
| `wait(seconds)` | Wait without acting |
| `game_over(reason)` | Signal game ended |

## v1 vs v2

| | v1 | v2 |
|---|---|---|
| **Vision** | OpenAI GPT-4o | Claude Sonnet |
| **Interaction** | Single-turn, coordinate grid | Multi-turn tool-use conversation |
| **Element detection** | None (guessed pixels) | OmniParser V2 (YOLOv8 + Florence-2) |
| **Accuracy** | ~50-120px error | Exact bounding box centers |
| **Multi-step** | Not possible | Natural (bid → PLAY in one cycle) |
| **Context** | None between actions | Full conversation history |
| **Cost** | Image scaling to 200px | Sliding window (8 images) |

## Known Limitations

- **Unity games** render on a single canvas — Android accessibility tools don't work, OmniParser's vision detection is the only option
- OmniParser adds ~3.5s latency per frame on Apple Silicon MPS
- `transformers` must be `<5` (v5 breaks Florence-2 custom code in OmniParser)
- Banner ads can occasionally steal taps
