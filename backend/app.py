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


# --- Persistence (disabled on serverless runtime) ---
def init_db():
    """Initialize persistence layer (disabled in this deployment mode)."""
    logger.info("Persistence disabled: running without local database writes")


def save_lead(email: str):
    """Accept lead data but do not persist it."""
    logger.info(f"[NO-PERSIST] Lead accepted: {email}")
    return False


def save_conversation(
    session_id: str, user_message: str, ai_response: str, sources: list
):
    """Accept conversation data but do not persist it."""
    _ = (session_id, user_message, ai_response, sources)
    logger.info("[NO-PERSIST] Conversation accepted")


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

    return jsonify(
        {
            "leads": [],
            "total": 0,
            "message": "Persistence is disabled in this deployment mode.",
        }
    )


@app.route("/api/admin/conversations", methods=["GET"])
def admin_conversations():
    """
    Admin endpoint — view all conversations.
    """
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")

    if token != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 403

    return jsonify(
        {
            "conversations": [],
            "total": 0,
            "message": "Persistence is disabled in this deployment mode.",
        }
    )


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
