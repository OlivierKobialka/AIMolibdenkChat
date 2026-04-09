"""
Lightweight RAG system for Moly AI Chat.
Uses local lexical retrieval (TF-IDF style scoring) + Gemini generation.
"""

import math
import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from threading import Lock

import google.generativeai as genai

from utils.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")
_index_lock = Lock()
_is_initialized = False


@dataclass
class Chunk:
    """In-memory indexed text chunk."""

    source: str
    text: str
    tf: Counter
    token_count: int


_chunks = []
_idf = {}


def _tokenize(text: str):
    return re.findall(r"\b[\w\-]+\b", text.lower(), flags=re.UNICODE)


def _split_text(text: str, chunk_size: int = 1400, overlap: int = 250):
    """Split long markdown content into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            split_idx = text.rfind("\n", start + 600, end)
            if split_idx == -1:
                split_idx = text.rfind(".", start + 600, end)
            if split_idx > start:
                end = split_idx + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks


def load_documents():
    """Load markdown documents from the data/documents directory."""
    docs_path = Path(DOCUMENTS_DIR)

    if not docs_path.exists():
        logger.error(f"Documents directory not found: {DOCUMENTS_DIR}")
        return []

    loaded = []
    for md_file in sorted(docs_path.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            for piece in _split_text(content):
                loaded.append((md_file.name, piece))
            logger.info(f"Loaded and split: {md_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {md_file.name}: {e}")

    logger.info(f"Total chunks loaded: {len(loaded)}")
    return loaded


def _build_index(raw_chunks):
    """Build an in-memory sparse index for fast lexical retrieval."""
    global _chunks, _idf

    indexed = []
    document_frequency = Counter()

    for source, text in raw_chunks:
        tokens = _tokenize(text)
        if not tokens:
            continue

        tf = Counter(tokens)
        indexed.append(Chunk(source=source, text=text, tf=tf, token_count=len(tokens)))
        document_frequency.update(tf.keys())

    if not indexed:
        _chunks = []
        _idf = {}
        return

    total_chunks = len(indexed)
    _idf = {
        term: math.log((1 + total_chunks) / (1 + df)) + 1.0
        for term, df in document_frequency.items()
    }
    _chunks = indexed


def _retrieve(question: str, k: int = 5):
    """Return top-k chunks relevant to the user question."""
    query_terms = _tokenize(question)
    if not query_terms or not _chunks:
        return []

    unique_terms = set(query_terms)
    scored = []
    for chunk in _chunks:
        score = 0.0
        for term in unique_terms:
            if term not in _idf:
                continue
            tf_norm = chunk.tf.get(term, 0) / max(chunk.token_count, 1)
            score += tf_norm * _idf[term]

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]


def _build_prompt(question: str, context_chunks):
    context = "\n\n".join(
        f"[Źródło: {chunk.source}]\n{chunk.text}" for chunk in context_chunks
    )
    return f"{SYSTEM_PROMPT}\n\n{RAG_PROMPT_TEMPLATE.format(context=context, question=question)}"


def _generate_answer(question: str, context_chunks):
    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY is not set")
        return "Brak konfiguracji AI (GOOGLE_API_KEY). Skontaktuj się z administratorem."

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    prompt = _build_prompt(question, context_chunks)

    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 2048,
        },
    )

    answer_text = getattr(response, "text", "")
    if answer_text:
        return answer_text.strip()

    return "Przepraszam, nie udało się wygenerować odpowiedzi."


def initialize_rag():
    """Initialize the lightweight RAG index once per process."""
    global _is_initialized

    with _index_lock:
        if _is_initialized:
            return

        logger.info("Initializing lightweight RAG index...")
        raw_chunks = load_documents()
        if not raw_chunks:
            raise RuntimeError("No documents found to load!")

        _build_index(raw_chunks)
        if not _chunks:
            raise RuntimeError("No valid document chunks were indexed!")

        _is_initialized = True
        logger.info(f"RAG ready with {_is_initialized} state and {len(_chunks)} chunks")


def ask_question(question: str) -> dict:
    """Ask a question using local retrieval + Gemini generation."""
    try:
        if not _is_initialized:
            initialize_rag()

        retrieved_chunks = _retrieve(question, k=5)
        if not retrieved_chunks:
            return {
                "answer": "Nie znalazłem odpowiedniego kontekstu w dokumentach. Jeśli chcesz, mogę przekierować pytanie do zespołu.",
                "sources": [],
            }

        answer = _generate_answer(question, retrieved_chunks)
        sources = []
        for chunk in retrieved_chunks:
            if chunk.source not in sources:
                sources.append(chunk.source)

        return {"answer": answer, "sources": sources}

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {
            "answer": "Przepraszam, wystąpił błąd podczas przetwarzania pytania. Proszę spróbować ponownie.",
            "sources": [],
        }
