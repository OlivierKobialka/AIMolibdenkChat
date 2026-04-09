"""
Moly AI Chat Assistant — Flask Backend
=======================================
Endpoints:
  POST /api/login    — Authenticate user (email + password)
  POST /api/chat     — Send question, get AI answer from documents
  GET  /api/admin/leads — View all registered leads (admin only)
  GET  /api/health   — Health check
"""

import os
import sqlite3
import logging
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment
load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# --- Flask App ---
app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [r"https://([a-zA-Z0-9-]+\.)*empiermedia\.com(:[0-9]+)?$"]
        }
    },
)

# --- Config ---
PASSWORD = os.getenv("APP_PASSWORD", "molibdenek2027!")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "moly-admin-secret-2027")
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "moly_chat.db")


# --- Database ---
def get_db():
    """Get SQLite database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            first_login_at TEXT NOT NULL,
            last_login_at TEXT NOT NULL,
            login_count INTEGER DEFAULT 1
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        )
    """
    )

    conn.commit()
    conn.close()
    logger.info("Database initialized")


def save_lead(email: str):
    """Save or update lead in database."""
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    cursor.execute("SELECT id, login_count FROM leads WHERE email = ?", (email,))
    existing = cursor.fetchone()

    is_new = False
    if existing:
        cursor.execute(
            "UPDATE leads SET last_login_at = ?, login_count = login_count + 1 WHERE email = ?",
            (now, email),
        )
    else:
        cursor.execute(
            "INSERT INTO leads (email, first_login_at, last_login_at, login_count) VALUES (?, ?, ?, 1)",
            (email, now, now),
        )
        is_new = True

    conn.commit()
    conn.close()
    return is_new


def save_conversation(
    session_id: str, user_message: str, ai_response: str, sources: list
):
    """Save conversation to database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (session_id, user_message, ai_response, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            session_id,
            user_message,
            ai_response,
            ",".join(sources),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


# ==============================
#  API ENDPOINTS
# ==============================


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "service": "Moly AI Chat",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/login", methods=["POST"])
def login():
    """
    Authenticate user with email + password.
    Password is hardcoded — same for all users.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Brak danych"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email:
        return jsonify({"error": "Email jest wymagany"}), 400

    if password != PASSWORD:
        return jsonify({"error": "Nieprawidłowe hasło"}), 401

    # Save lead
    is_new = save_lead(email)

    # Send email notification for new users (async-safe, non-blocking)
    if is_new:
        try:
            from utils.email_sender import send_new_user_notification

            send_new_user_notification(email)
        except Exception as e:
            logger.error(f"Email notification failed: {e}")

    logger.info(f"User logged in: {email} (new={is_new})")

    return jsonify(
        {"success": True, "session_id": email, "message": "Zalogowano pomyślnie"}
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Handle chat messages. Uses RAG to answer from documents.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Brak danych"}), 400

    session_id = data.get("session_id", "")
    message = data.get("message", "").strip()

    if not session_id:
        return jsonify({"error": "Brak sesji. Proszę się zalogować."}), 401

    if not message:
        return jsonify({"error": "Wiadomość nie może być pusta"}), 400

    # Query RAG system
    try:
        from utils.rag import ask_question

        result = ask_question(message)

        answer = result["answer"]
        sources = result["sources"]

        # Save conversation
        save_conversation(session_id, message, answer, sources)

        return jsonify({"response": answer, "sources": sources})

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return (
            jsonify(
                {"error": "Przepraszam, wystąpił błąd. Proszę spróbować ponownie."}
            ),
            500,
        )


@app.route("/api/admin/leads", methods=["GET"])
def admin_leads():
    """
    Admin endpoint — view all leads.
    Requires Authorization header with admin token.
    """
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")

    if token != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM leads ORDER BY last_login_at DESC")
    rows = cursor.fetchall()
    conn.close()

    leads = []
    for row in rows:
        leads.append(
            {
                "id": row["id"],
                "email": row["email"],
                "first_login_at": row["first_login_at"],
                "last_login_at": row["last_login_at"],
                "login_count": row["login_count"],
            }
        )

    return jsonify({"leads": leads, "total": len(leads)})


@app.route("/api/admin/conversations", methods=["GET"])
def admin_conversations():
    """
    Admin endpoint — view all conversations.
    """
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")

    if token != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations ORDER BY created_at DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()

    conversations = []
    for row in rows:
        conversations.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "user_message": row["user_message"],
                "ai_response": row["ai_response"],
                "sources": row["sources"],
                "created_at": row["created_at"],
            }
        )

    return jsonify({"conversations": conversations, "total": len(conversations)})


# ==============================
#  APP STARTUP
# ==============================

if __name__ == "__main__":
    # Initialize database
    init_db()

    # Initialize RAG pipeline
    try:
        from utils.rag import initialize_rag

        logger.info("Starting RAG initialization...")
        initialize_rag()
        logger.info("RAG pipeline ready!")
    except Exception as e:
        logger.error(f"RAG initialization failed: {e}")
        logger.warning("Chat will not work until RAG is initialized.")

    # Run Flask
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"Starting Moly AI Chat Backend on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
