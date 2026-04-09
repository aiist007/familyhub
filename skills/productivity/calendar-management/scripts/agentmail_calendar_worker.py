#!/usr/bin/env python3
"""
Agentmail Calendar Worker
- Polls agentmail.to inbox for new unread messages
- Daytime (07:00-21:00): every 10 min
- Nighttime (21:00-07:00): every 60 min
- Extracts calendar JSON via Nemotron-3-super-120b-a12b
- Creates Google Calendar event
- Marks thread as read
- Only processes messages with subject containing: hotel, booking, 予約, チェックイン,
  check-in, flight, フライト, meeting, ミーティング, 予定, appointment, event

Usage:
  python agentmail_calendar_worker.py [--once]
"""

import json, os, sys, time, re, urllib.request, urllib.error
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/root/.hermes"))
VENV_PYTHON = HERMES_HOME / "hermes-agent/venv/bin/python3"
SCRIPT_DIR  = Path(__file__).parent
STATE_FILE  = SCRIPT_DIR / ".agentmail_calendar_state.json"

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

NVIDIA_KEY     = ENV.get("NVIDIA_API_KEY", "")
AGENTMAIL_KEY  = ENV.get("AGENTMAIL_API_KEY", "")
AGENTMAIL_BASE = "https://api.agentmail.to/v0"
NV_API         = "https://integrate.api.nvidia.com/v1"
BOT_TOKEN      = ENV.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API   = f"https://api.telegram.org/bot{BOT_TOKEN}"
TELEGRAM_CHAT  = ENV.get("TELEGRAM_CALENDAR_CHAT_ID", "")

# ── Family Config ───────────────────────────────────────────────────────────
FAMILY_CONFIG = SCRIPT_DIR / ".family_config.json"

def load_family():
    if FAMILY_CONFIG.exists():
        return json.loads(FAMILY_CONFIG.read_text())
    return {"members": [], "setup_done": False}

family = load_family()

# Keywords that indicate calendar-relevant content
CAL_KEYWORDS = [
    "hotel", "booking", "予約", "チェックイン", "check-in", "checkin",
    "flight", "フライト", "meeting", "ミーティング", "予定", "appointment",
    "event", "到着", "出発", "departure", "arrival", "check-out", "checkout",
    "restaurant", "餐厅", "饭店", "dinner", "lunch", "seminar", "conference"
]

# ── State ───────────────────────────────────────────────────────────────────
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed_thread_ids": []}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state))

# ── NVIDIA LLM (fallback → nous) ─────────────────────────────────────────────
NOUS_API  = "https://inference-api.nousresearch.com/v1"
NOUS_KEY  = ENV.get("NOUS_API_KEY", "")

def _call_llm(base_url: str, api_key: str, model: str, messages: list[dict], max_tokens: int = 512) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
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

def nv_chat(model: str, messages: list[dict], max_tokens: int = 512) -> str:
    """Primary: NVIDIA. Fallback: nous/xiaomi/mimo-v2-pro."""
    # Try NVIDIA first
    try:
        return _call_llm(NV_API, NVIDIA_KEY, model, messages, max_tokens)
    except Exception as e:
        print(f"[WARN] NVIDIA failed ({e}), falling back to nous")

    # Fallback to nous
    if not NOUS_KEY:
        raise RuntimeError("NVIDIA failed and NOUS_API_KEY not set for fallback")
    return _call_llm(NOUS_API, NOUS_KEY, "xiaomi/mimo-v2-pro", messages, max_tokens)

# ── Calendar JSON Generator ─────────────────────────────────────────────────
def build_cal_prompt():
    base = """You are a calendar event JSON generator for a family. Given email text, extract structured calendar information and identify which family member this event belongs to.

Output format (Google Calendar native):
{
  "summary": "Event title",
  "start": {"dateTime": "YYYY-MM-DDTHH:MM:SS+08:00"},
  "end": {"dateTime": "YYYY-MM-DDTHH:MM:SS+08:00"},
  "location": "Address or link",
  "description": "Details",
  "extendedProperties": {
    "private": {
      "people": "Participants",
      "owner": "Family member name or null if unknown"
    }
  }
}

Rules:
- Output ONLY JSON, no markdown, no explanation
- start/end must include time in HH:MM:SS format
- Use Asia/Shanghai (UTC+8) unless text specifies Japan (Asia/Tokyo = UTC+9)
- For hotel check-in: start=check-in date 15:00, end=check-out date 11:00
- For flights: start=departure datetime, end=arrival datetime
- Match owner to family members based on email recipient, subject, or content context
- If unclear, set owner to null"""

    members = family.get("members", [])
    if members:
        base += "\n\nFAMILY MEMBERS:\n"
        for m in members:
            aliases = ", ".join(m.get("aliases", []))
            base += f'- {m["name"]} (age {m.get("age", "?")}, aliases: {aliases if aliases else "none"})\n'
        base += "\nMatch 'owner' to one of the above. If unclear, set owner to null."
    return base

def extract_calendar_json(text: str) -> dict:
    for attempt in range(2):
        try:
            content = nv_chat(
                "nvidia/nemotron-3-super-120b-a12b",
                [{"role": "system", "content": build_cal_prompt()},
                 {"role": "user", "content": text},
                 {"role": "assistant", "content": "{"}],
                max_tokens=512,
                temperature=0
            )
            content = "{" + (content or "").strip()
            if len(content) < 5:
                continue
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
                return json.loads(content[start:end])
        except Exception as e:
            print(f"[WARN] extract attempt {attempt+1} failed: {e}")
    return None

# ── GCal Event Field Helpers ────────────────────────────────────────────────
def _evt_dt(event: dict, field: str) -> str:
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

def match_member_from_text(text: str) -> dict | None:
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

def send_telegram_question(text: str):
    """Push a question to Telegram chat (family calendar management)."""
    if not BOT_TOKEN or not TELEGRAM_CHAT:
        print(f"[WARN] Telegram not configured, skipping push notification")
        return
    payload = {"chat_id": TELEGRAM_CHAT, "text": text, "parse_mode": "Markdown"}
    req = urllib.request.Request(
        f"{TELEGRAM_API}/sendMessage", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode()
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
        print(f"[TELEGRAM] Sent question to chat {TELEGRAM_CHAT}")
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")

# ── Google Calendar Create ──────────────────────────────────────────────────
def create_gcal_event(event: dict) -> dict:
    """Create GCal event from Google Calendar native JSON format."""
    import subprocess
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

# ── Agentmail API ────────────────────────────────────────────────────────────
def list_inboxes():
    req = urllib.request.Request(
        f"{AGENTMAIL_BASE}/inboxes",
        headers={"Authorization": f"Bearer {AGENTMAIL_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read()).get("inboxes", [])

def list_threads(inbox_id: str, limit: int = 20):
    req = urllib.request.Request(
        f"{AGENTMAIL_BASE}/inboxes/{inbox_id}/messages?limit={limit}",
        headers={"Authorization": f"Bearer {AGENTMAIL_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read()).get("messages", [])

def get_thread(thread_id: str):
    req = urllib.request.Request(
        f"{AGENTMAIL_BASE}/threads/{thread_id}",
        headers={"Authorization": f"Bearer {AGENTMAIL_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())

def mark_thread_read(thread_id: str):
    payload = json.dumps({"add_labels": ["read"], "remove_labels": ["unread"]}).encode()
    req = urllib.request.Request(
        f"{AGENTMAIL_BASE}/threads/{thread_id}",
        method="PATCH",
        headers={"Authorization": f"Bearer {AGENTMAIL_KEY}", "Content-Type": "application/json"},
        data=payload
    )
    with urllib.request.urlopen(req, timeout=10):
        pass

# ── Filtering ────────────────────────────────────────────────────────────────
def is_calendar_relevant(subject: str, text: str) -> bool:
    combined = (subject + " " + text).lower()
    return any(kw in combined for kw in CAL_KEYWORDS)

# ── Main message handler ─────────────────────────────────────────────────────
def process_thread(thread_id: str, state: dict) -> bool:
    """Returns True if a calendar event was created."""
    if thread_id in state["processed_thread_ids"]:
        return False

    try:
        thread = get_thread(thread_id)
    except Exception as e:
        print(f"[ERROR] Failed to get thread {thread_id}: {e}")
        return False

    msgs = thread.get("messages", [])
    if not msgs:
        return False

    msg = msgs[0]
    subject = msg.get("subject", "")
    text = msg.get("text", "")[:4000]  # limit for API

    if not is_calendar_relevant(subject, text):
        print(f"[SKIP] Not calendar-relevant: {subject[:60]}")
        return False

    print(f"[PROCESS] thread={thread_id} subject={subject[:60]}")

    try:
        event = extract_calendar_json(text)
        print(f"[JSON] {json.dumps(event, ensure_ascii=False)}")
    except Exception as e:
        print(f"[ERROR] JSON extraction failed: {e}")
        return False

    if not event or not event.get("summary"):
        print(f"[SKIP] No calendar data extracted")
        return False

    # ── Owner identification ───────────────────────────────────────────────
    owner_name = ""
    owner = _evt_owner(event)
    if owner and family.get("members"):
        matched = match_member_from_text(owner)
        if matched:
            owner_name = matched["name"]
            ext = event.setdefault("extendedProperties", {}).setdefault("private", {})
            ext["owner"] = owner_name
            desc = event.get("description", "")
            event["description"] = f"[主人: {owner_name}] {desc}".strip()

    if not owner_name and family.get("members"):
        # Can't determine owner → push to Telegram to ask
        member_names = ", ".join(m["name"] for m in family["members"])
        q = (
            f"📧 邮件日程无法确认归属人：\n\n"
            f"📌 事情: {event.get('summary')}\n"
            f"🕐 时间: {_evt_dt(event,'start')} → {_evt_dt(event,'end')}\n"
            f"📍 地点: {event.get('location','TBD')}\n"
            f"📄 来源: {subject}\n\n"
            f"❓ 这是谁的日程？（{member_names}）"
        )
        send_telegram_question(q)
        # Still create the event (without owner), mark as processed
        desc = event.get("description", "")
        event["description"] = f"[主人: 待确认] {desc}".strip()

    # ── Create calendar event ──────────────────────────────────────────────
    try:
        gcal = create_gcal_event(event)
        print(f"[GCAL] Created: {gcal.get('htmlLink', '')}")
        mark_thread_read(thread_id)
        state["processed_thread_ids"].append(thread_id)
        if len(state["processed_thread_ids"]) > 500:
            state["processed_thread_ids"] = state["processed_thread_ids"][-500:]
        save_state(state)
        return True
    except Exception as e:
        print(f"[ERROR] GCal create failed: {e}")
        return False

# ── Polling interval ───────────────────────────────────────────────────────
def get_poll_interval() -> int:
    """Return sleep seconds based on current hour (Asia/Shanghai).
    07:00-21:00 → 600s (10 min)
    21:00-07:00 → 3600s (60 min)
    """
    from datetime import datetime, timezone, timedelta
    tz_cn = timezone(timedelta(hours=8))
    hour = datetime.now(tz_cn).hour
    if 7 <= hour < 21:
        return 600   # 10 minutes
    return 3600      # 1 hour

# ── Main loop ────────────────────────────────────────────────────────────────
def main():
    run_once = "--once" in sys.argv
    state = load_state()

    if not AGENTMAIL_KEY:
        print("[ERROR] AGENTMAIL_API_KEY not set")
        sys.exit(1)

    while True:
        try:
            inboxes = list_inboxes()
            for inbox in inboxes:
                inbox_id = inbox.get("inbox_id")
                if not inbox_id:
                    continue
                threads = list_threads(inbox_id, limit=10)
                unread = [t for t in threads if "unread" in t.get("labels", [])]
                print(f"[CHECK] inbox={inbox_id} unread={len(unread)}")
                for t in unread:
                    process_thread(t.get("thread_id"), state)
        except Exception as e:
            print(f"[ERROR] Main loop error: {e}")

        if run_once:
            print("[DONE] --once mode, exiting.")
            break
        interval = get_poll_interval()
        print(f"[SLEEP] {interval}s (next check ~{interval//60}min)")
        time.sleep(interval)

if __name__ == "__main__":
    main()
