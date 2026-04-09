#!/usr/bin/env python3
"""
Telegram Interactive Agent
- Polls Telegram for new messages (text + photo)
- Intent classification: calendar extraction vs general chat
- Calendar: OCR → JSON → Google Calendar event
- Chat: conversational reply with context memory
- Replies via Telegram bot

Usage:
  python telegram_calendar_agent.py [--once]
"""

import json, os, sys, time, re, tempfile, urllib.request, urllib.error
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/root/.hermes"))
VENV_PYTHON = HERMES_HOME / "hermes-agent/venv/bin/python3"
SCRIPT_DIR  = Path(__file__).parent
STATE_FILE  = SCRIPT_DIR / ".telegram_calendar_state.json"

# ── Env ─────────────────────────────────────────────────────────────────────
def load_env():
    env = {}
    for line in open(HERMES_HOME / ".env"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

ENV = load_env()

BOT_TOKEN    = ENV.get("TELEGRAM_BOT_TOKEN", "")
ALLOWED      = set(ENV.get("TELEGRAM_ALLOWED_USERS", "").split(","))
NVIDIA_KEY   = ENV.get("NVIDIA_API_KEY", "")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
NV_API       = "https://integrate.api.nvidia.com/v1"
AGENTMAIL_KEY = ENV.get("AGENTMAIL_API_KEY", "")

# ── Family Config ───────────────────────────────────────────────────────────
FAMILY_CONFIG = SCRIPT_DIR / ".family_config.json"

def load_family():
    if FAMILY_CONFIG.exists():
        return json.loads(FAMILY_CONFIG.read_text())
    return {"members": [], "setup_done": False}

def save_family(config):
    FAMILY_CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2))

family = load_family()

# ── State ───────────────────────────────────────────────────────────────────
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_update_id": None}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state))

# ── NVIDIA LLM (fallback → nous) ─────────────────────────────────────────────
NOUS_API  = "https://inference-api.nousresearch.com/v1"
NOUS_KEY  = ENV.get("NOUS_API_KEY", "")

def _call_llm(base_url: str, api_key: str, model: str, messages: list[dict], max_tokens: int = 512, temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload).encode()
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read())
        return d["choices"][0]["message"]["content"]

def nv_chat(model: str, messages: list[dict], max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Primary: NVIDIA. Fallback: nous/xiaomi/mimo-v2-pro."""
    try:
        return _call_llm(NV_API, NVIDIA_KEY, model, messages, max_tokens, temperature)
    except Exception as e:
        print(f"[WARN] NVIDIA failed ({e}), falling back to nous")
    if not NOUS_KEY:
        raise RuntimeError("NVIDIA failed and NOUS_API_KEY not set for fallback")
    return _call_llm(NOUS_API, NOUS_KEY, "xiaomi/mimo-v2-pro", messages, max_tokens, temperature)

# ── Conversation History ───────────────────────────────────────────────────
CHAT_HISTORY: dict[int, list[dict]] = {}  # chat_id → [{"role":..., "content":...}]
MAX_HISTORY = 20  # messages per chat

def add_history(chat_id: int, role: str, content: str):
    if chat_id not in CHAT_HISTORY:
        CHAT_HISTORY[chat_id] = []
    CHAT_HISTORY[chat_id].append({"role": role, "content": content})
    if len(CHAT_HISTORY[chat_id]) > MAX_HISTORY:
        CHAT_HISTORY[chat_id] = CHAT_HISTORY[chat_id][-MAX_HISTORY:]

# ── Intent Classifier ─────────────────────────────────────────────────────
INTENT_PROMPT = """Classify the user message. Reply with ONE word only:
- "calendar" if the message contains scheduling info, dates/times for events, meeting invitations, booking confirmations, travel itineraries, reminders, or asks to add something to a calendar.
- "chat" for everything else (questions, conversation, requests, greetings, etc).

Reply ONLY "calendar" or "chat"."""

def classify_intent(text: str) -> str:
    """Returns 'calendar' or 'chat'."""
    try:
        result = nv_chat(
            "nvidia/nemotron-3-super-120b-a12b",
            [{"role": "system", "content": INTENT_PROMPT}, {"role": "user", "content": text}],
            max_tokens=10
        )
        result = result.strip().lower()
        if "calendar" in result:
            return "calendar"
        return "chat"
    except Exception:
        return "chat"  # default to chat on failure

# ── Chat Response ──────────────────────────────────────────────────────────
CHAT_SYSTEM = """You are a calendar management assistant on Telegram, connected to the user's Google Calendar.

Key facts:
- You CAN read and write to the user's Google Calendar (it's already connected)
- You create calendar events automatically when schedule info is detected
- You can query upcoming events when asked
- You are helping manage the SAME Google Calendar that's integrated in the system

Reply rules:
- Reply in the same language the user uses (Chinese → Chinese, English → English)
- Keep responses under 200 words
- Be practical, no fluff
- NEVER say you can't access the calendar — you CAN, it's already connected"""

def generate_chat_reply(chat_id: int, user_text: str, ocr_text: str = "") -> str:
    """Generate a conversational reply."""
    messages = [{"role": "system", "content": CHAT_SYSTEM}]
    # Add history
    messages.extend(CHAT_HISTORY.get(chat_id, []))
    # Add current message
    content = user_text
    if ocr_text:
        content += f"\n\n[Image text extracted via OCR]:\n{ocr_text}"
    messages.append({"role": "user", "content": content})

    try:
        reply = nv_chat("nvidia/nemotron-3-super-120b-a12b", messages, max_tokens=1024)
    except Exception:
        reply = nv_chat("xiaomi/mimo-v2-pro", messages, max_tokens=1024)
    return reply

# ── OCR via Nemotron Nano VL ────────────────────────────────────────────────
def ocr_image(image_url: str) -> str:
    """Use Nemotron Nano VL to extract text from image.
    Downloads image to local temp file, then passes as base64 data URL
    because NVIDIA API cannot fetch Telegram file URLs directly."""
    import base64, tempfile

    # Download image
    try:
        with urllib.request.urlopen(image_url, timeout=30) as resp:
            img_data = resp.read()
        img_ext = image_url.rsplit(".", 1)[-1] if "." in image_url else "jpeg"
        mime = "image/png" if img_ext == "png" else "image/jpeg"
        b64 = base64.b64encode(img_data).decode()
        data_url = f"data:{mime};base64,{b64}"
        print(f"[IMG] Downloaded {len(img_data)} bytes, mime={mime}")
    except Exception as e:
        print(f"[ERROR] Image download failed: {e}")
        return ""

    ocr_prompt = (
        "This image is likely a Chinese chat screenshot, calendar screenshot, or booking confirmation. "
        "请逐字仔细阅读图片中的所有文字内容，包括：中文、英文、数字、标点、emoji、发送者名字、时间戳。 "
        "不要遗漏任何文字。逐字输出你能看到的所有内容，保留原始换行。"
    )
    payload = {
        "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": ocr_prompt}
            ]}
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        f"{NV_API}/chat/completions",
        method="POST",
        headers={
            "Authorization": f"Bearer {NVIDIA_KEY}",
            "Content-Type": "application/json"
        },
        data=json.dumps(payload).encode()
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            d = json.loads(resp.read())
            return d["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300] if e.fp else ""
        print(f"[ERROR] OCR API error: {e.code} — {body}")
        raise
    except Exception as e:
        print(f"[ERROR] OCR error: {e}")
        raise

# ── Calendar JSON Generator ─────────────────────────────────────────────────
def build_cal_prompt():
    """Build calendar extraction prompt with family member context."""
    base = """You are a family calendar event JSON generator. Extract schedule information and identify which family member this event belongs to.

REQUIRED output format (Google Calendar native):
{
  "summary": "Event title",
  "start": {"dateTime": "YYYY-MM-DDTHH:MM:SS+08:00"},
  "end": {"dateTime": "YYYY-MM-DDTHH:MM:SS+08:00"},
  "location": "Where",
  "description": "Details and inference notes",
  "extendedProperties": {
    "private": {
      "people": "Participants (comma separated)",
      "owner": "Family member name or null if unknown"
    }
  }
}

FOUR ESSENTIAL ELEMENTS — each must be filled:
1. TIME (start/end): When? Use context clues to infer.
2. LOCATION (location): Where? "TBD" if unknown.
3. PEOPLE (people): Who? From OCR names, chat mentions. "TBD" if unknown.
4. EVENT (summary): What is happening?

FIFTH ELEMENT — owner identification:
5. OWNER: Which family member does this event belong to? Match based on:
   - Direct mention of their name, alias, or role (e.g. "爸爸", "太太", specific name)
   - Context clues (e.g. "宝宝打疫苗" → child, "老婆出差" → wife)
   - If unsure, set owner to null (do NOT guess)

If the text has NO calendar/scheduling info at all, output:
{"summary": null}

INFERENCE RULES:
- Current datetime context will be provided — use it to resolve relative dates
- "明天" = tomorrow, "后天" = day after tomorrow, "下周X" = next weekday X
- "下午3点" = 15:00, "上午" = morning (09:00 default), "晚上" = evening (19:00 default)
- "周末" = coming Saturday/Sunday
- If only date without time: meetings 09:00, meals 12:00, check-in 15:00
- Hotel: check-in 15:00, check-out next day 11:00
- From chat screenshots: look for names in message bubbles, group chat participant names
- description should note HOW time/location was inferred

Output ONLY JSON, no markdown, no explanation."""

    members = family.get("members", [])
    if members:
        member_desc = "\n\nFAMILY MEMBERS:\n"
        for m in members:
            aliases = ", ".join(m.get("aliases", []))
            member_desc += f'- {m["name"]} (age {m.get("age", "?")}, aliases: {aliases if aliases else "none"})\n'
        member_desc += "\nMatch 'owner' to one of the above members. If unclear, set owner to null."
        base += member_desc
    return base

def extract_calendar_json(text: str, context: str = "") -> dict:
    """Extract calendar event JSON. context = current datetime + conversation history."""
    from datetime import datetime, timezone, timedelta
    tz_cn = timezone(timedelta(hours=8))
    now = datetime.now(tz_cn).strftime("%Y-%m-%d %H:%M:%S %A")
    user_msg = f"[Current datetime: {now}]\n"
    if context:
        user_msg += f"[Recent conversation context:\n{context}]\n"
    user_msg += f"\n--- Message to analyze ---\n{text}"

    for attempt in range(2):
        try:
            content = nv_chat(
                "nvidia/nemotron-3-super-120b-a12b",
                [{"role": "system", "content": build_cal_prompt()},
                 {"role": "user", "content": user_msg},
                 {"role": "assistant", "content": "{"}],
                max_tokens=512,
                temperature=0
            )
            # Prepend the { we seeded
            content = "{" + (content or "").strip()
            if len(content) < 5:
                print(f"[WARN] Too short on attempt {attempt+1}")
                continue
            # Find outermost JSON object
            depth = 0
            start = content.find("{")
            if start == -1:
                continue
            end = -1
            for i, c in enumerate(content[start:], start):
                if c == "{": depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                json_str = content[start:end]
                return json.loads(json_str)
            print(f"[WARN] No balanced JSON on attempt {attempt+1}")
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON parse failed on attempt {attempt+1}: {e}, content={content[:300]!r}")
        except Exception as e:
            print(f"[WARN] extract_calendar attempt {attempt+1} failed: {e}")
    return None

# ── GCal Event Field Helpers ────────────────────────────────────────────────
def _evt_dt(event: dict, field: str) -> str:
    """Extract datetime string from GCal native start/end field."""
    val = event.get(field, {})
    if isinstance(val, str):
        return val
    return val.get("dateTime", val.get("date", "?"))

def _evt_people(event: dict) -> str:
    ext = event.get("extendedProperties", {}).get("private", {})
    return ext.get("people", event.get("people", "TBD"))

def _evt_owner(event: dict) -> str | None:
    ext = event.get("extendedProperties", {}).get("private", {})
    return ext.get("owner", event.get("owner"))

# ── Google Calendar Create ──────────────────────────────────────────────────
def create_gcal_event(event: dict) -> dict:
    """Create GCal event from Google Calendar native JSON format."""
    import subprocess
    # Extract owner/people from extendedProperties for description enrichment
    ext = event.get("extendedProperties", {}).get("private", {})
    owner = ext.get("owner", "")
    people = ext.get("people", "")
    desc = event.get("description", "")
    if owner:
        desc = f"[主人: {owner}] {desc}".strip()
    if people and people != "TBD":
        desc = f"参与人: {people}\n{desc}" if desc else f"参与人: {people}"
    if desc:
        event["description"] = desc
    # Pass native GCal JSON directly
    cmd = [
        str(VENV_PYTHON),
        str(HERMES_HOME / "skills/productivity/google-workspace/scripts/google_api.py"),
        "calendar", "create",
        "--json", json.dumps(event, ensure_ascii=False),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"GCal create failed: {result.stderr}")
    return json.loads(result.stdout)

# ── Telegram: short poll ────────────────────────────────────────────────────
def poll_telegram(state: dict) -> list:
    """Short poll (timeout=0) — does NOT conflict with gateway long-polling."""
    offset = state["last_update_id"]
    url = f"{TELEGRAM_API}/getUpdates?timeout=0&limit=10"
    if offset is not None:
        url += f"&offset={offset}"

    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            d = json.loads(resp.read())
            if not d.get("ok"):
                return []
            updates = d.get("result", [])
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("[ERROR] Invalid bot token")
        elif e.code == 409:
            print("[WARN] Conflict — gateway is polling. Sleep 30s.")
            time.sleep(30)
        return []
    except Exception as e:
        print(f"[WARN] Poll error: {e}")
        return []

    return updates

def send_reply(chat_id: int, text: str):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    req = urllib.request.Request(
        f"{TELEGRAM_API}/sendMessage",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode()
    )
    with urllib.request.urlopen(req, timeout=10):
        pass

def get_photo_url(file_id: str) -> str:
    """Get direct file URL from file_id."""
    req = urllib.request.Request(f"{TELEGRAM_API}/getFile?file_id={file_id}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        d = json.loads(resp.read())
        path = d["result"]["file_path"]
    return f"https://api.telegram.org/file/bot{BOT_TOKEN}/{path}"

# ── Google Calendar Query ─────────────────────────────────────────────────
def query_gcal_events(date_from: str = "", date_to: str = "") -> list:
    """Query calendar events. date_from/date_to in ISO format."""
    import subprocess
    cmd = [
        str(VENV_PYTHON),
        str(HERMES_HOME / "skills/productivity/google-workspace/scripts/google_api.py"),
        "calendar", "list",
    ]
    if date_from:
        cmd.extend(["--start", date_from])
    if date_to:
        cmd.extend(["--end", date_to])
    cmd.extend(["--max", "10"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"[ERROR] GCal query failed: {result.stderr}")
            return []
        return json.loads(result.stdout) if result.stdout.strip() else []
    except Exception as e:
        print(f"[ERROR] GCal query error: {e}")
        return []

def format_events_reply(events: list) -> str:
    """Format calendar events for Telegram reply."""
    if not events:
        return "📅 没有找到日程安排。"
    lines = [f"📅 找到 {len(events)} 个日程:\n"]
    for i, ev in enumerate(events[:10], 1):
        # Handle both dict and string inputs
        if isinstance(ev, str):
            ev = json.loads(ev) if ev.startswith("{") else {"summary": ev}
        start_raw = ev.get("start", {})
        end_raw = ev.get("end", {})
        if isinstance(start_raw, str):
            start = start_raw
        else:
            start = start_raw.get("dateTime", start_raw.get("date", "?"))
        if isinstance(end_raw, str):
            end = end_raw
        else:
            end = end_raw.get("dateTime", end_raw.get("date", ""))
        summary = ev.get("summary", "(无标题)")
        location = ev.get("location", "")
        desc = ev.get("description", "")
        # Extract people from description
        people = ""
        if desc and "参与人:" in desc:
            people_line = [l for l in desc.split("\n") if "参与人:" in l]
            if people_line:
                people = people_line[0].replace("参与人:", "").strip()
        # Format time nicely
        start_fmt = start[:16].replace("T", " ") if len(start) > 10 else start
        lines.append(f"{i}. 📌 {summary}")
        lines.append(f"   🕐 {start_fmt}")
        if location:
            lines.append(f"   📍 {location}")
        if people:
            lines.append(f"   👥 {people}")
        lines.append("")
    return "\n".join(lines)

# ── Intent: add calendar vs query calendar vs chat ────────────────────────
QUERY_PROMPT = """Classify the user message. Reply with ONE word:
- "query" if asking about schedule, calendar, events (e.g. "what's my schedule", "do I have anything tomorrow", "what meetings", "我明天有什么安排", "看下日程")
- "modify" if wanting to change, cancel, reschedule an existing event (e.g. "move that to 3pm", "cancel the meeting", "改到下周", "取消")
- "add" if describing a new event to add (e.g. "meeting with John at 3pm", "book flight on Friday", "明天下午跟张总吃饭")
- "chat" for everything else

Reply ONLY one word: query, modify, add, or chat."""

def classify_calendar_intent(text: str) -> str:
    try:
        result = nv_chat(
            "nvidia/nemotron-3-super-120b-a12b",
            [{"role": "system", "content": QUERY_PROMPT}, {"role": "user", "content": text}],
            max_tokens=10
        )
        result = result.strip().lower()
        for w in ["query", "modify", "add", "chat"]:
            if w in result:
                return w
        return "add"  # default to trying to add
    except Exception:
        return "add"

# ── Mark agentmail thread as read ─────────────────────────────────────────
def mark_agentmail_read(thread_id: str):
    if not AGENTMAIL_KEY:
        return
    BASE = "https://api.agentmail.to/v0"
    payload = json.dumps({"add_labels": ["read"], "remove_labels": ["unread"]}).encode()
    req = urllib.request.Request(
        f"{BASE}/threads/{thread_id}",
        method="PATCH",
        headers={"Authorization": f"Bearer {AGENTMAIL_KEY}", "Content-Type": "application/json"},
        data=payload
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as e:
        print(f"[WARN] Agentmail mark read failed: {e}")

# ── Setup state tracking ────────────────────────────────────────────────────
SETUP_STATE: dict[int, dict] = {}  # chat_id → {"step": "name"|"age"|"aliases"|"done", "current": {...}}
PENDING_EVENTS: dict[int, dict] = {}  # chat_id → {"event": {...}, "gcal_result": None}

def match_member_from_text(text: str) -> dict | None:
    """Try to match text (user reply or message context) to a family member."""
    members = family.get("members", [])
    if not members:
        return None
    text_lower = text.lower().strip()
    for m in members:
        if m["name"].lower() in text_lower:
            return m
        for alias in m.get("aliases", []):
            if alias.lower() in text_lower:
                return m
    return None

def handle_setup(chat_id: int, text: str) -> str:
    """First-time family setup flow. Returns reply text."""
    global family
    members = family.get("members", [])

    # Skip setup with "skip" or "跳过"
    if text.lower() in ("skip", "跳过", "不用了", "no"):
        family["setup_done"] = True
        save_family(family)
        SETUP_STATE.pop(chat_id, None)
        return "✅ 好的，跳过家庭成员设置。稍后可以重新设置。"

    # Quick setup: "J 38岁 alias1,alias2; 太太 35岁 alias3,alias4"
    if text and ";" in text and text.count("岁") > 0:
        for part in text.split(";"):
            part = part.strip()
            if not part:
                continue
            # Parse: "name age aliases..."
            m = re.match(r"(\S+)\s+(\d+)\s*岁?\s*(.*)", part)
            if m:
                name = m.group(1)
                age = int(m.group(2))
                aliases_str = m.group(3).strip()
                aliases = [a.strip() for a in aliases_str.split(",") if a.strip()] if aliases_str else []
                members.append({"name": name, "age": age, "calendar_id": None, "aliases": aliases})
        family["members"] = members
        family["setup_done"] = True
        save_family(family)
        SETUP_STATE.pop(chat_id, None)
        names = ", ".join(m["name"] for m in members)
        return f"✅ 家庭成员已设置: {names}\n\n现在开始管理家庭日程！"

    # Step-by-step setup
    step = SETUP_STATE.get(chat_id, {}).get("step", "start")

    if step == "start":
        SETUP_STATE[chat_id] = {"step": "name", "current": {}}
        return (
            "👋 欢迎使用家庭日程管理！\n\n"
            "请设置家庭成员信息。逐个添加成员，格式：\n"
            "名字 年龄 别名1,别名2\n\n"
            "例如：J 38岁 张总,爸爸\n\n"
            "或者一次性输入，用分号分隔：\n"
            "J 38岁 张总,爸爸; 太太 35岁 老婆,Mrs.J; 宝宝 5岁 小宝\n\n"
            "输入「跳过」可暂不设置。"
        )

    if step == "name":
        SETUP_STATE[chat_id] = {"step": "confirm", "current": {}}
        # Parse single member entry
        m = re.match(r"(\S+)\s+(\d+)\s*岁?\s*(.*)", text)
        if m:
            name = m.group(1)
            age = int(m.group(2))
            aliases_str = m.group(3).strip()
            aliases = [a.strip() for a in aliases_str.split(",") if a.strip()] if aliases_str else []
            member = {"name": name, "age": age, "calendar_id": None, "aliases": aliases}
            members.append(member)
            family["members"] = members
            save_family(family)
            aliases_display = ", ".join(aliases) if aliases else "无"
            return (
                f"✅ 已添加: {name} ({age}岁, 别名: {aliases_display})\n\n"
                f"继续添加下一位成员，或输入「完成」结束设置。"
            )
        else:
            return "格式不对，请按「名字 年龄 别名1,别名2」输入，例如：J 38岁 张总,爸爸"

    if step == "confirm":
        if text.lower() in ("完成", "done", "ok", "好了"):
            family["setup_done"] = True
            save_family(family)
            SETUP_STATE.pop(chat_id, None)
            names = ", ".join(m["name"] for m in members)
            return f"✅ 家庭成员设置完成: {names}\n\n开始管理家庭日程吧！"
        # Otherwise treat as another member entry
        SETUP_STATE[chat_id] = {"step": "name", "current": {}}
        return handle_setup(chat_id, text)

    return "请输入「名字 年龄 别名」或「完成」"

# ── Main message handler ─────────────────────────────────────────────────────
def process_message(update: dict, state: dict):
    global family
    msg = update.get("message") or update.get("edited_message") or update.get("channel_post")
    if not msg:
        return

    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    chat_type = chat.get("type", "")

    # Text content
    text = msg.get("text") or msg.get("caption") or ""

    # Photo
    photos = msg.get("photo", [])
    has_image = bool(photos)

    # Check allowlist (skip if not in allowed users)
    from_id = str(msg.get("from", {}).get("id", ""))
    if ALLOWED and from_id not in ALLOWED and chat_type != "channel":
        print(f"[SKIP] chat_id={chat_id} not in allowlist")
        return

    print(f"[MSG] chat_id={chat_id} text={text[:50]!r} has_image={has_image}")

    # ── First-time setup flow ───────────────────────────────────────────────
    if not family.get("setup_done"):
        reply = handle_setup(chat_id, text.strip())
        send_reply(chat_id, reply)
        add_history(chat_id, "user", text[:200] if text else "[photo]")
        add_history(chat_id, "assistant", reply[:500])
        return

    # ── Handle pending event (user answering "which member?") ───────────────
    if chat_id in PENDING_EVENTS and text and not has_image:
        pending = PENDING_EVENTS.pop(chat_id)
        event = pending["event"]
        # Match user reply to a family member
        matched = match_member_from_text(text.strip())
        if matched:
            # Store owner in extendedProperties
            ext = event.setdefault("extendedProperties", {}).setdefault("private", {})
            ext["owner"] = matched["name"]
            desc = event.get("description", "")
            event["description"] = f"[主人: {matched['name']}] {desc}".strip()
            try:
                gcal = create_gcal_event(event)
                reply = (
                    f"✅ *日程已添加* ({matched['name']})\n\n"
                    f"📌 事情: {event.get('summary')}\n"
                    f"🕐 时间: {_evt_dt(event,'start')} → {_evt_dt(event,'end')}\n"
                    f"📍 地点: {event.get('location','TBD')}\n"
                    f"👥 人物: {_evt_people(event)}\n\n"
                    f"[查看日历]({gcal.get('htmlLink','')})"
                )
            except Exception as e:
                reply = f"⚠️ 创建失败: {e}\n```\n{json.dumps(event, ensure_ascii=False)}\n```"
        else:
            # User said something but we couldn't match - just add without owner
            try:
                gcal = create_gcal_event(event)
                reply = (
                    f"✅ *日程已添加*\n\n"
                    f"📌 事情: {event.get('summary')}\n"
                    f"🕐 时间: {_evt_dt(event,'start')} → {_evt_dt(event,'end')}\n"
                    f"📍 地点: {event.get('location','TBD')}\n\n"
                    f"[查看日历]({gcal.get('htmlLink','')})"
                )
            except Exception as e:
                reply = f"⚠️ 创建失败: {e}"
        send_reply(chat_id, reply)
        add_history(chat_id, "user", text[:200])
        add_history(chat_id, "assistant", reply[:300])
        return

    # OCR if image present (with retry)
    ocr_text = ""
    if has_image:
        for attempt in range(2):
            try:
                file_id = photos[-1]["file_id"]
                photo_url = get_photo_url(file_id)
                print(f"[IMG] OCR attempt {attempt+1}: {photo_url}")
                ocr_text = ocr_image(photo_url)
                print(f"[OCR] Result: {ocr_text[:200]!r}")
                if ocr_text:
                    break
            except Exception as e:
                print(f"[ERROR] OCR attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    time.sleep(2)

    # Build combined text — prioritize text message, supplement with OCR
    if text and ocr_text and ocr_text != "NO_TEXT_IN_IMAGE":
        combined_text = f"{text}\n\n--- 截图文字 ---\n{ocr_text}"
    elif ocr_text and ocr_text != "NO_TEXT_IN_IMAGE":
        combined_text = ocr_text
    elif text:
        combined_text = text  # OCR failed but we still have user's text
    else:
        print("[SKIP] No text content")
        return

    # ── Build context for calendar extraction ──
    history_lines = []
    for h in CHAT_HISTORY.get(chat_id, [])[-6:]:  # last 6 exchanges
        role = "用户" if h["role"] == "user" else "助手"
        history_lines.append(f"{role}: {h['content'][:100]}")
    conv_context = "\n".join(history_lines) if history_lines else ""

    # ── Classify intent: query / modify / add / chat ──
    intent = classify_calendar_intent(combined_text)
    print(f"[INTENT] {intent}")

    reply = None

    if intent == "query":
        # ── Query calendar ──
        from datetime import datetime, timezone, timedelta
        tz_cn = timezone(timedelta(hours=8))
        now = datetime.now(tz_cn)
        # Smart date range based on query
        date_from = now.strftime("%Y-%m-%dT00:00:00+08:00")
        date_to = (now + timedelta(days=7)).strftime("%Y-%m-%dT23:59:59+08:00")
        if "明天" in combined_text:
            date_from = (now + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00+08:00")
            date_to = (now + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59+08:00")
        elif "今天" in combined_text or "今日" in combined_text:
            pass  # already today
        elif "这周" in combined_text or "本周" in combined_text:
            date_to = (now + timedelta(days=(6 - now.weekday()))).strftime("%Y-%m-%dT23:59:59+08:00")
        elif "下周" in combined_text:
            days_ahead = 7 - now.weekday()
            date_from = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT00:00:00+08:00")
            date_to = (now + timedelta(days=days_ahead + 6)).strftime("%Y-%m-%dT23:59:59+08:00")
        print(f"[QUERY] {date_from} → {date_to}")
        events = query_gcal_events(date_from, date_to)
        reply = format_events_reply(events)

    elif intent == "modify":
        # ── Modify — chat-based for now ──
        reply = generate_chat_reply(chat_id,
            f"[User wants to modify a calendar event. Help them.]\n{text}",
            ocr_text if ocr_text != "NO_TEXT_IN_IMAGE" else "")

    else:  # "add" or "chat" — always try to extract calendar first
        # ── Add calendar event ──
        event = None
        gcal_result = None
        try:
            event = extract_calendar_json(combined_text, context=conv_context)
            if event:
                print(f"[JSON] {json.dumps(event, ensure_ascii=False)}")
            else:
                print("[INFO] No calendar info extracted (model returned null)")
        except Exception as e:
            print(f"[INFO] Calendar extraction error: {e}")
            event = None

        # If calendar event found → check owner → create it
        if event and event.get("summary"):
            owner = _evt_owner(event)
            owner_name = ""
            if owner and family.get("members"):
                matched = match_member_from_text(owner)
                if matched:
                    owner_name = matched["name"]
                    ext = event.setdefault("extendedProperties", {}).setdefault("private", {})
                    ext["owner"] = owner_name
                    desc = event.get("description", "")
                    event["description"] = f"[主人: {owner_name}] {desc}".strip()

            if owner_name:
                # Owner identified → create event
                try:
                    gcal = create_gcal_event(event)
                    print(f"[GCAL] Created for {owner_name}: {gcal.get('htmlLink', '')}")
                    reply = (
                        f"✅ *日程已添加* ({owner_name})\n\n"
                        f"📌 事情: {event.get('summary')}\n"
                        f"🕐 时间: {_evt_dt(event,'start')} → {_evt_dt(event,'end')}\n"
                        f"📍 地点: {event.get('location','TBD')}\n"
                        f"👥 人物: {_evt_people(event)}\n\n"
                        f"[查看日历]({gcal.get('htmlLink','')})"
                    )
                except Exception as e:
                    reply = f"⚠️ 创建失败: {e}\n```\n{json.dumps(event, ensure_ascii=False)}\n```"
            elif owner is None and family.get("members"):
                # Owner unknown → ask user
                member_names = ", ".join(m["name"] for m in family["members"])
                PENDING_EVENTS[chat_id] = {"event": event}
                reply = (
                    f"📋 已识别日程：\n\n"
                    f"📌 事情: {event.get('summary')}\n"
                    f"🕐 时间: {_evt_dt(event,'start')} → {_evt_dt(event,'end')}\n"
                    f"📍 地点: {event.get('location','TBD')}\n\n"
                    f"❓ 这是谁的日程？（{member_names}）"
                )
            else:
                # No family members configured → create without owner
                try:
                    gcal = create_gcal_event(event)
                    print(f"[GCAL] Created: {gcal.get('htmlLink', '')}")
                    reply = (
                        f"✅ *日程已添加*\n\n"
                        f"📌 事情: {event.get('summary')}\n"
                        f"🕐 时间: {_evt_dt(event,'start')} → {_evt_dt(event,'end')}\n"
                        f"📍 地点: {event.get('location','TBD')}\n"
                        f"👥 人物: {_evt_people(event)}\n\n"
                        f"[查看日历]({gcal.get('htmlLink','')})"
                    )
                except Exception as e:
                    reply = f"⚠️ 创建失败: {e}\n```\n{json.dumps(event, ensure_ascii=False)}\n```"
        elif event and event.get("summary"):
            reply = f"⚠️ 已识别日程但创建失败:\n```\n{json.dumps(event, ensure_ascii=False)}\n```"
        else:
            # No calendar info → chat reply
            reply = generate_chat_reply(chat_id, text, ocr_text if ocr_text != "NO_TEXT_IN_IMAGE" else "")

    # Save to conversation history
    user_content = combined_text[:500]  # cap to avoid huge history
    add_history(chat_id, "user", user_content)
    add_history(chat_id, "assistant", reply[:500])

    # Send reply
    try:
        send_reply(chat_id, reply)
    except Exception as e:
        print(f"[ERROR] Telegram reply failed: {e}")

# ── Main loop ────────────────────────────────────────────────────────────────
def main():
    run_once = "--once" in sys.argv
    state = load_state()

    while True:
        updates = poll_telegram(state)

        for upd in updates:
            update_id = upd.get("update_id", 0)
            process_message(upd, state)
            # Advance offset past this update
            if update_id >= (state["last_update_id"] or 0):
                state["last_update_id"] = update_id + 1
                save_state(state)

        if run_once or not updates:
            if run_once:
                print("[DONE] --once mode, exiting.")
                break
            time.sleep(10)  # short idle before next poll

if __name__ == "__main__":
    main()
