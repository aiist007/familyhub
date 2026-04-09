---
name: calendar-management
description: "Unified calendar management system: Telegram interactive agent (OCR+chat+query+add) and AgentMail email worker (auto-extract from inbox). Supports Chinese dates, 4-element events (time/location/people/event), context-aware inference."
version: 1.0.0
author: Hermes Agent
license: MIT
required_environment_variables:
  - TELEGRAM_BOT_TOKEN
  - TELEGRAM_ALLOWED_USERS
  - AGENTMAIL_API_KEY
  - NVIDIA_API_KEY
  - NOUS_API_KEY
  - TELEGRAM_CALENDAR_CHAT_ID  # chat_id for AgentMail → Telegram push (your numeric chat ID)
---

# Calendar Management System

Unified calendar assistant with two ingestion channels:

1. **Telegram Agent** — interactive chat bot, handles images via OCR, extracts calendar events, queries schedule
2. **AgentMail Worker** — polls email inbox, auto-extracts calendar events from booking confirmations

Both write to Google Calendar via `google_api.py calendar create`.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Google Calendar                 │
│         (single source of truth)                 │
└────────────┬──────────────────┬──────────────────┘
             │                  │
    ┌────────┴───────┐  ┌──────┴────────┐
    │ Telegram Agent │  │ AgentMail Wkr  │
    │ (interactive)  │  │ (background)   │
    └────────┬───────┘  └──────┬────────┘
             │                  │
    ┌────────┴───────┐  ┌──────┴────────┐
    │ User messages  │  │ Email inbox   │
    │ text + images  │  │ familyhub@... │
    └────────────────┘  └───────────────┘
```

### Data Flow

```
INPUT (text/image/email)
  ↓
OCR (if image) → Nemotron Nano VL
  ↓
Intent: query / modify / add / chat
  ├─ query  → GCal list → format reply
  ├─ modify → chat-based modification
  ├─ add    → extract 4 elements → GCal create
  └─ chat   → conversational reply
  ↓
REPLY / CONFIRMATION
```

## Components

### 1. Telegram Interactive Agent

**File:** `scripts/telegram_calendar_agent.py`

Features:
- Short-polling (`timeout=0`) — no conflict with Hermes gateway
- 4 intents: query, modify, add, chat
- OCR with retry (2 attempts) + text fallback
- 4 calendar elements: time, location, people, event
- Context inference: current datetime + conversation history
- Chinese date support: "明天", "下周三", "下午3点"
- Per-chat conversation history (20 messages, in-memory)

### 2. AgentMail Email Worker

**File:** `scripts/agentmail_calendar_worker.py`

Features:
- Smart polling: daytime (07-21) every 10min, nighttime (21-07) every 60min
- Keyword filter: hotel, booking, flight, meeting, etc.
- Auto-extract calendar from email body
- Mark thread as read after processing
- Dedup via processed thread IDs (state file, max 500)

### Calendar JSON Format

Both components extract Google Calendar native format:

```json
{
  "summary": "Event title",
  "start": {"dateTime": "YYYY-MM-DDTHH:MM:SS+08:00"},
  "end": {"dateTime": "YYYY-MM-DDTHH:MM:SS+08:00"},
  "location": "Where",
  "description": "Details, booking ref, inference notes",
  "extendedProperties": {
    "private": {
      "people": "Participants (comma separated)",
      "owner": "Family member name or null"
    }
  }
}
```

This JSON is passed directly to `google_api.py calendar create --json '...'`, which sends it as-is to Google Calendar API. `people` and `owner` are stored in `extendedProperties.private` (custom metadata, not displayed in GCal UI).

### Four Essential Elements

| Element | Field | Rules |
|---------|-------|-------|
| TIME | start.dateTime / end.dateTime | Inferred from context if relative |
| LOCATION | location | "TBD" if unknown |
| PEOPLE | extendedProperties.private.people | From OCR names, chat mentions |
| EVENT | summary | Concise title |

### Context Inference

`extract_calendar_json(text, context)` receives:
- Current datetime (auto-injected, Asia/Shanghai)
- Recent conversation history (Telegram) or email subject (AgentMail)

Resolves: "明天" → concrete date, "跟张总" → people="张总"

## Models

| Model | Provider | Purpose |
|-------|----------|---------|
| nvidia/nemotron-3-super-120b-a12b | NVIDIA NGC | Calendar JSON extraction, intent classification, chat |
| nvidia/llama-3.1-nemotron-nano-vl-8b-v1 | NVIDIA NGC | OCR (image → text) |
| xiaomi/mimo-v2-pro | Nous | Fallback for all NVIDIA calls |

### Fallback Chain

```
NVIDIA API → Nous API (xiaomi/mimo-v2-pro)
```

`nv_chat()` tries NVIDIA first, falls back to Nous on failure.

## Environment Setup

### Required API Keys

```bash
# .env file (~/.hermes/.env)
TELEGRAM_BOT_TOKEN=<bot_token>
TELEGRAM_ALLOWED_USERS=<numeric_user_id>
AGENTMAIL_API_KEY=<agentmail_key>
NVIDIA_API_KEY=nvapi-...
NOUS_API_KEY=<nous_key>  # extract from auth.json, ~24h TTL
```

### Extract NOUS_API_KEY

```bash
python3 -c "
import json
d = json.load(open('/root/.hermes/auth.json'))
key = d['providers']['nous']['agent_key']
print(f'NOUS_API_KEY={key}')
"
```

### Google Calendar Auth

Requires `google_token.json` in HERMES_HOME with Calendar API scope.

## Deployment

### Start Both Services

```bash
cd ~/.hermes && source hermes-agent/venv/bin/activate

# Telegram Agent
PYTHONUNBUFFERED=1 nohup python scripts/telegram_calendar_agent.py \
  > /tmp/telegram_calendar_agent.log 2>&1 &

# AgentMail Worker
PYTHONUNBUFFERED=1 nohup python scripts/agentmail_calendar_worker.py \
  > /tmp/agentmail_calendar_worker.log 2>&1 &
```

### Stop

```bash
pkill -f telegram_calendar_agent.py
pkill -f agentmail_calendar_worker.py
```

### One-Shot Test

```bash
python scripts/telegram_calendar_agent.py --once
python scripts/agentmail_calendar_worker.py --once
```

### Monitor Logs

```bash
tail -f /tmp/telegram_calendar_agent.log
tail -f /tmp/agentmail_calendar_worker.log
```

## Telegram Bot Commands

| User sends | Bot action |
|------------|------------|
| "19号下午2点香港见张总" | Extract 4 elements → GCal → confirm |
| [screenshot] + "明天这个" | OCR + caption → GCal → confirm |
| "看下明天日程" | Query GCal → list events |
| "这周有什么安排" | Query GCal → list this week |
| "改到下周三" | Chat-based modification help |
| "今天天气怎么样" | General chat reply |

## AgentMail Polling Schedule

| Time | Interval |
|------|----------|
| 07:00 - 21:00 | 10 minutes |
| 21:00 - 07:00 | 60 minutes |

## State Files

| File | Content |
|------|---------|
| `scripts/.telegram_calendar_state.json` | `{last_update_id: N}` |
| `scripts/.agentmail_calendar_state.json` | `{processed_thread_ids: [...]}` |

State is persisted across restarts. Telegram state prevents re-processing old messages. AgentMail state prevents duplicate calendar events.

## OCR Image Download Fix

**Root cause:** NVIDIA API returns HTTP 500 when given Telegram file URLs directly (`https://api.telegram.org/file/bot...`). The API cannot fetch external URLs.

**Solution:** Download the image to local memory first, base64-encode it, pass as a data URL:

```python
import base64, urllib.request

# Download from Telegram
with urllib.request.urlopen(photo_url, timeout=30) as resp:
    img_data = resp.read()
b64 = base64.b64encode(img_data).decode()
data_url = f"data:image/jpeg;base64,{b64}"

# Pass data URL to NVIDIA VL model
payload = {
    "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": ocr_prompt}
    ]}],
}
```

## OCR Chinese Text Optimization

The default OCR prompt doesn't work well for Chinese chat screenshots. Use this prompt instead:

```python
ocr_prompt = (
    "This image is likely a Chinese chat screenshot, calendar screenshot, or booking confirmation. "
    "请逐字仔细阅读图片中的所有文字内容，包括：中文、英文、数字、标点、emoji、发送者名字、时间戳。 "
    "不要遗漏任何文字。逐字输出你能看到的所有内容，保留原始换行。"
)
```

## Family Member Support

The system supports multi-member family calendar management with per-member event ownership.

### First-Time Setup

When `setup_done` is false, the bot guides the user through setup on first message:

**Quick setup (one line):**
```
J 38岁 张总,爸爸; 太太 35岁 老婆,Mrs.J; 宝宝 5岁 小宝
```

**Step-by-step:** send any message and the bot will prompt for member info.

### Config File

`scripts/.family_config.json`:
```json
{
  "members": [
    {
      "name": "J",
      "age": 38,
      "calendar_id": null,
      "aliases": ["张总", "爸爸"]
    }
  ],
  "setup_done": true
}
```

### Owner Identification Flow

```
Extract event → model returns "owner" field
  ├─ owner matches a member → create with [主人: name]
  ├─ owner is null + members configured → ask user
  │   └─ Telegram: "❓ 这是谁的日程？（J, 太太, 宝宝）"
  │   └─ AgentMail: push to TELEGRAM_CALENDAR_CHAT_ID
  └─ no members configured → create without owner
```

### AgentMail → Telegram Push

AgentMail worker pushes questions to Telegram when it can't determine the member. Requires:
- `TELEGRAM_CALENDAR_CHAT_ID` in .env (your numeric Telegram chat ID)
- `TELEGRAM_BOT_TOKEN` configured

## Watchdog Auto-Restart

`deploy.sh start` now includes a watchdog process that checks every 30 seconds:

```bash
bash deploy.sh start    # starts both agents + watchdog
bash deploy.sh stop     # stops everything including watchdog
bash deploy.sh status   # shows watchdog status too
```

The watchdog runs as `bash deploy.sh watchdog` in background via nohup.

## Google Calendar Integration

The system uses the **same Google Calendar** that's already connected in Hermes (`google_token.json`). Both Telegram Agent and AgentMail Worker write to this calendar via `google_api.py calendar create`. The bot knows it has full calendar access and will never say it can't access the calendar.

## Migration Checklist

To deploy on a new system:

1. [ ] Copy scripts:
   - `scripts/telegram_calendar_agent.py`
   - `scripts/agentmail_calendar_worker.py`

2. [ ] Set environment variables in `.env`:
   - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USERS`
   - `AGENTMAIL_API_KEY`
   - `NVIDIA_API_KEY`, `NOUS_API_KEY`

3. [ ] Ensure Google Calendar auth:
   - `google_token.json` with Calendar scope
   - `google_api.py` script available

4. [ ] Install dependencies (stdlib only — no pip packages needed):
   - Python 3.10+
   - urllib, json, os, sys, time, re, subprocess, pathlib

5. [ ] Start services with `nohup` or systemd

6. [ ] Verify with `--once` mode and test messages

## JSON Extraction Technique

The calendar extraction uses an assistant-seeded prompt to force JSON output:
- Messages: `[system: CAL_PROMPT, user: text, assistant: "{"]`
- The assistant prefix `{}` forces the model to continue as JSON
- `temperature=0` for deterministic output
- Balanced brace parser extracts the outermost `{...}` object

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| OCR 500 error | Wrong model ID | Use `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` (NOT `nemotron-nano-12b-v2-vl`) |
| Telegram 409 Conflict | Gateway long-polling conflict | Use `timeout=0` short poll |
| Bot not responding | User ID not in allowlist | Add numeric user ID to `TELEGRAM_ALLOWED_USERS` |
| GCal create fails | Auth expired | Refresh `google_token.json` |
| NVIDIA 404 | Invalid provider config | Do NOT set `HERMES_INFERENCE_PROVIDER=nvidia` |
| Fallback chain broken | `fallback_model` in config.yaml | Remove it; use `fallback_providers: []` |
| NOUS API 401 | Key expired (24h TTL) | Re-extract from `auth.json` |
| AgentMail timeout | API slow | Increase timeout in script (default 15s) |
| Empty log file | Python output buffering | Add `PYTHONUNBUFFERED=1` |
| No calendar extracted | Model returned empty/null | Check `[WARN]` logs; retry with better text |
