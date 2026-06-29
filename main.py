import time

_boot_start = time.time()
print(f"[BOOT] main.py import started at {_boot_start}")

import hashlib
import mimetypes
import os
import re
import threading
from contextlib import asynccontextmanager

from auth import (
    create_access_token,
    get_current_user,
    require_admin,
    require_admin_or_operator,
    verify_password,
)
from database import DocumentMetadata, SessionLocal, User, get_db, init_db

# Import custom application modules
from extractor import extract_text_from_file
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from rag_engine import (
    analyze_document,
    delete_document_from_vector_store,
    inject_text_into_vector_store,
    query_document_intelligence,
)
from sqlalchemy import func
from sqlalchemy.orm import Session, load_only


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    from rag_engine import warm_up

    loop = asyncio.get_event_loop()
    # Removed init_db and warm_up from lifespan to prevent Render port binding timeout
    # Tables should be created via migrate.py or a pre-deploy command.
    yield


app = FastAPI(title="CCL DocIntel API Engine", lifespan=lifespan)
print(f"[BOOT] FastAPI ready in {time.time() - _boot_start:.2f}s")

# 2. Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    # Allow local dev (http/https), any *.onrender.com, and any *.vercel.app deployment
    allow_origin_regex=r"(https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.onrender\.com|https://.*\.vercel\.app)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StripLocaleMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            match = re.match(r"^/[a-zA-Z]{2}(-[a-zA-Z]{2})?(/.*)", path)
            if match:
                scope["path"] = match.group(2)
                if "raw_path" in scope:
                    scope["raw_path"] = match.group(2).encode("ascii")
        await self.app(scope, receive, send)


app.add_middleware(StripLocaleMiddleware)

# Set up local temporary document repository directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "documents")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


# Data Validation Models
class ChatQuery(BaseModel):
    query_text: str
    history: list[dict] = []


class LoginRequest(BaseModel):
    username: str
    password: str


# ─────────────────────────────────────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/api/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """Validates credentials and returns a signed JWT token."""
    username = payload.username.strip().lower()
    # Friendly alias: many users type "view" when they mean the viewer account.
    if username == "view":
        username = "viewer"

    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(payload.password.strip(), user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    if not user.is_active:
        raise HTTPException(
            status_code=403, detail="Account is disabled. Contact your administrator."
        )
    token = create_access_token(data={"sub": user.username, "role": user.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "username": user.username,
    }


@app.get("/api/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Returns the currently authenticated user's info."""
    return current_user


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/wake")
def wake_server():
    """
    Pre-warm endpoint called by the frontend on page load.
    Kicks off PGVector + embedding engine initialization in a background
    thread so the first real chat request isn't hitting a cold Python
    import path. Returns immediately so the wakeup loader is fast.
    """
    import threading

    def _warm():
        try:
            from rag_engine import get_embedding_engine, get_vector_db

            get_embedding_engine()
            get_vector_db()
            print("[WAKE] Vector DB and embeddings pre-warmed.")
        except Exception as e:
            print(f"[WAKE] Pre-warm skipped: {e}")

    threading.Thread(target=_warm, daemon=True).start()
    return {"status": "awake"}


@app.get("/")
def read_root():
    return {
        "status": "Online",
        "system": "CCL DocIntel Core Engine",
        "database": "Neon Cloud PostgreSQL Connected",
    }


@app.post("/api/chat")
def handle_chat(payload: ChatQuery, current_user: dict = Depends(get_current_user)):
    """RAG chat endpoint — open to all authenticated users."""
    ai_response_text = query_document_intelligence(
        payload.query_text, payload.history, user_role=current_user["role"]
    )
    return {"sender": "bot", "text": ai_response_text}


@app.get("/api/analytics/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    total_docs = db.query(func.count(DocumentMetadata.id)).scalar() or 0
    total_chars = db.query(func.sum(DocumentMetadata.char_count)).scalar() or 0
    vectorized_nodes = int(total_chars / 200) if total_chars > 0 else 0
    active_flags = (
        db.query(func.count(DocumentMetadata.id))
        .filter(DocumentMetadata.risk_level.in_(["High", "Medium"]))
        .scalar()
        or 0
    )
    if total_docs > 0:
        integrity_score = round((1 - (active_flags / total_docs)) * 100, 1)
    else:
        integrity_score = 100.0
    return {
        "vectorizedNodes": vectorized_nodes,
        "activeFlags": active_flags,
        "complianceIntegrity": f"{integrity_score}%",
    }


@app.get("/api/compliance/alerts")
def get_compliance_alerts(db: Session = Depends(get_db)):
    """Fetches AI-flagged compliance violations."""
    try:
        flagged_docs = (
            db.query(DocumentMetadata)
            .options(
                load_only(
                    DocumentMetadata.id,
                    DocumentMetadata.filename,
                    DocumentMetadata.risk_level,
                    DocumentMetadata.risk_description,
                )
            )
            .filter(DocumentMetadata.risk_level.in_(["High", "Medium"]))
            .order_by(DocumentMetadata.upload_date.desc())
            .all()
        )
        alerts = []
        for doc in flagged_docs:
            alerts.append(
                {
                    "id": doc.id,
                    "doc": doc.filename,
                    "type": f"{doc.risk_level} Risk Rule Violation",
                    "risk": doc.risk_level,
                    "desc": doc.risk_description,
                }
            )
        if not alerts:
            return [
                {
                    "id": 0,
                    "doc": "System Stable",
                    "type": "Operational Stability",
                    "risk": "None",
                    "desc": "No active vulnerabilities flagged across repository nodes.",
                }
            ]
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
@app.post("/api/documents")
def list_uploaded_documents(
    skip: int = 0,
    limit: int = 5,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Returns a paginated slice of documents.
    NOTE: Auto-pruning has been REMOVED because Render uses an ephemeral filesystem.
    Physical files are lost on every restart, but extracted_text lives in the cloud
    PostgreSQL DB and must be preserved. Delete via the DELETE endpoint instead.
    """
    # Use load_only to exclude the large blob columns (raw_file_data, extracted_text)
    # from the list query — they are not needed for the document cards.
    _list_cols = load_only(
        DocumentMetadata.id,
        DocumentMetadata.filename,
        DocumentMetadata.content_type,
        DocumentMetadata.char_count,
        DocumentMetadata.upload_date,
        DocumentMetadata.access_level,
        DocumentMetadata.department,
        DocumentMetadata.doc_type,
        DocumentMetadata.summary,
        DocumentMetadata.risk_level,
        DocumentMetadata.risk_description,
    )
    base_query = db.query(DocumentMetadata).options(_list_cols)
    if current_user["role"] != "admin":
        base_query = base_query.filter(
            DocumentMetadata.access_level.in_(["Public", "Internal"])
        )

    total_count = base_query.count()
    documents = (
        base_query.order_by(DocumentMetadata.upload_date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return {
        "total": total_count,
        "data": [
            {
                "id": d.id,
                "filename": d.filename,
                "contentType": d.content_type,
                "char_count": d.char_count,
                "uploadDate": d.upload_date.isoformat() if d.upload_date else None,
                "access_level": d.access_level,
                "department": d.department,
                "doc_type": d.doc_type,
                "summary": d.summary,
                "risk_level": d.risk_level,
                "risk_description": d.risk_description,
            }
            for d in documents
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN-ONLY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

import threading

# Limit concurrent processing to prevent Render Free Tier Out-Of-Memory (OOM 512MB limit)
# 1 at a time prevents two heavy LLM+embedding calls competing for the same 512MB heap.
_ai_semaphore = threading.Semaphore(1)
_vector_semaphore = threading.Semaphore(1)


def _run_vector_injection(parsed_text: str, filename: str, access_level: str):
    """Phase 2: Vector injection runs AFTER badge is updated (non-blocking)."""
    with _vector_semaphore:
        try:
            inject_text_into_vector_store(parsed_text, filename, access_level)
            print(f"[BG] Vector injection done for '{filename}'.")
        except Exception as e:
            print(f"[BG] Vector injection failed for '{filename}': {e}")


def _run_ai_background(
    doc_id: int, filename: str, access_level: str = "Internal", text: str = None
):
    """
    Phase 1: Extract text + AI scan + update DB badge immediately.
    Phase 2: Vector injection fires in its own thread so badge updates FAST.

    `text` – pre-supplied text (e.g. after a manual edit via update_document_text).
             When provided, disk reading is skipped entirely so the edited text
             is never overwritten by the original file content.
             When None, the file is read from disk with a DB fallback for
             Render restarts where the ephemeral filesystem was wiped.
    """
    try:
        with _ai_semaphore:
            if text is not None:
                # Use caller-supplied text (manual edit path)
                parsed_text = text
            else:
                file_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.exists(file_path):
                    parsed_text = extract_text_from_file(file_path)
                else:
                    # Render ephemeral disk was wiped — fall back to stored DB text
                    print(
                        f"[BG] File '{filename}' missing from disk; using stored text."
                    )
                    _db = SessionLocal()
                    try:
                        _doc = (
                            _db.query(DocumentMetadata)
                            .filter(DocumentMetadata.id == doc_id)
                            .first()
                        )
                        parsed_text = (_doc.extracted_text or "") if _doc else ""
                    finally:
                        _db.close()

            char_count = len(parsed_text)
            analysis_result = analyze_document(parsed_text, filename)

        db = SessionLocal()
        try:
            doc = (
                db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
            )
            if doc:
                doc.char_count = char_count
                # Only overwrite extracted_text when we actually read it ourselves
                # (i.e., text was not pre-supplied by the caller)
                if text is None:
                    doc.extracted_text = parsed_text[:50000]
                doc.risk_level = analysis_result["risk_level"]
                doc.risk_description = analysis_result["description"]
                doc.department = analysis_result.get("department", "Unknown")
                doc.doc_type = analysis_result.get("doc_type", "Unknown")
                doc.summary = analysis_result.get("summary", "")
                db.commit()
                print(
                    f"[BG] Badge updated for '{filename}': {analysis_result['risk_level']}"
                )
        finally:
            db.close()

        # Phase 2: Vector injection in separate thread - does NOT block badge update
        threading.Thread(
            target=_run_vector_injection,
            args=(parsed_text, filename, access_level),
            daemon=True,
        ).start()

    except Exception as e:
        import traceback

        err_msg = traceback.format_exc()
        print(f"[BG] Background AI task failed for '{filename}': {e}")
        db = SessionLocal()
        try:
            doc = (
                db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
            )
            if doc:
                doc.risk_level = "Error"
                # Only store the crash summary — never overwrite the document's text
                doc.risk_description = f"Scan failed: {str(e)[:200]}"
                db.commit()
        except:
            pass
        finally:
            db.close()


@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    force: bool = Form(False),
    access_level: str = Form("Internal"),
    db: Session = Depends(get_db),
    _user: dict = Depends(require_admin_or_operator),  # 🔒 ADMIN + OPERATOR
):
    try:
        content = await file.read()

        # ── 1. Check for filename conflicts ──
        existing_by_name = (
            db.query(DocumentMetadata)
            .filter(DocumentMetadata.filename == file.filename)
            .first()
        )
        if existing_by_name:
            raise HTTPException(
                status_code=409,
                detail=f"A file with the name '{file.filename}' already exists in the system.",
            )

        # ── 2. Save to disk ──
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # ── 3. Fast Duplicate Detection (Raw File Hash) ──
        file_hash = hashlib.sha256(content).hexdigest()
        existing_by_hash = (
            db.query(DocumentMetadata)
            .filter(DocumentMetadata.content_hash == file_hash)
            .first()
        )

        if existing_by_hash:
            if not force:
                os.remove(file_path)
                raise HTTPException(
                    status_code=409,
                    detail={
                        "type": "similar_content",
                        "existing_filename": existing_by_hash.filename,
                        "message": f"This file contains the same data as '{existing_by_hash.filename}'.",
                    },
                )

        # ── 4. Fix Content-Type ──
        actual_content_type = file.content_type
        if not actual_content_type or actual_content_type == "application/octet-stream":
            guessed_type, _ = mimetypes.guess_type(file.filename)
            if guessed_type:
                actual_content_type = guessed_type

        # ── 5. Save a "pending" DB record immediately ──
        db_record = DocumentMetadata(
            filename=file.filename,
            content_type=actual_content_type,
            char_count=0,
            extracted_text="",
            risk_level="Scanning...",
            risk_description="AI compliance scan in progress.",
            content_hash=file_hash,
            access_level=access_level,
            raw_file_data=content,
        )
        db.add(db_record)
        db.commit()
        # No db.refresh() needed — SQLAlchemy populates db_record.id during flush.
        # Calling refresh() with large LargeBinary columns would read the full
        # file back from PostgreSQL into RAM unnecessarily.

        # ── 6. Offload ALL slow work (text extraction + AI) to background thread ──
        background_tasks.add_task(
            _run_ai_background, db_record.id, file.filename, access_level
        )

        # ── 7. Return immediately ──
        return {
            "status": "Success",
            "filename": db_record.filename,
            "characterCount": 0,
            "previewText": "",
            "message": "File saved. AI compliance scan running in background.",
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}/text")
@app.post("/api/documents/{doc_id}/text")
def get_document_text(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Returns the stored extracted text for a document (for the View modal)."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if current_user["role"] != "admin" and doc.access_level == "Confidential":
        raise HTTPException(
            status_code=403, detail="Access denied. Confidential document."
        )
    return {
        "id": doc.id,
        "filename": doc.filename,
        "char_count": doc.char_count,
        "risk_level": doc.risk_level,
        "risk_description": doc.risk_description,
        "extracted_text": doc.extracted_text
        or "No text content available for this file.",
    }


@app.get("/api/documents/{doc_id}/file")
@app.post("/api/documents/{doc_id}/file")
def get_document_file(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Returns the physical file for native viewing (PDF/Images)."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if current_user["role"] != "admin" and doc.access_level == "Confidential":
        raise HTTPException(
            status_code=403, detail="Access denied. Confidential document."
        )
    # Return from database if available (this survives Render restarts)
    if getattr(doc, "raw_file_data", None) is not None:
        headers = {}
        if doc.content_type == "application/pdf":
            headers["Content-Disposition"] = f'inline; filename="{doc.filename}"'
        return Response(
            content=doc.raw_file_data, media_type=doc.content_type, headers=headers
        )

    # Fallback to local disk (might be wiped by Render, but keep just in case)
    file_path = os.path.join(UPLOAD_DIR, doc.filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Physical file missing from disk and no backup in database.",
        )

    headers = {}
    if doc.content_type == "application/pdf":
        headers["Content-Disposition"] = f'inline; filename="{doc.filename}"'

    return FileResponse(
        file_path, media_type=doc.content_type, filename=doc.filename, headers=headers
    )


@app.delete("/api/documents/{doc_id}")
def delete_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin),  # 🔒 ADMIN ONLY
):
    """Deletes a document record from DB, disk, and vector store."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # 1. Remove from vector store (in background to make UI instant)
    try:
        from rag_engine import delete_document_from_vector_store

        background_tasks.add_task(delete_document_from_vector_store, doc.filename)
    except Exception:
        pass

    # 2. Remove physical file
    file_path = os.path.join(UPLOAD_DIR, doc.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # 3. Delete DB record
    db.delete(doc)
    db.commit()
    return {"status": "Deleted", "filename": doc.filename}


class UpdateDocumentRequest(BaseModel):
    text: str


@app.put("/api/documents/{doc_id}/text")
async def update_document_text(
    doc_id: int,
    payload: UpdateDocumentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin),
):
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # 1. Update text and reset risk status
    doc.extracted_text = payload.text
    doc.char_count = len(payload.text)
    doc.risk_level = "Scanning..."
    doc.risk_description = "AI compliance scan in progress after text edit."

    # Optional: Update content_hash based on new text
    hash_material = payload.text.encode("utf-8")
    doc.content_hash = hashlib.sha256(hash_material).hexdigest()

    db.commit()

    # 2. Re-run AI analysis on the NEW text (pass text directly so the
    #    background task never re-reads from disk and overwrites the edit)
    background_tasks.add_task(
        _run_ai_background,
        doc.id,
        doc.filename,
        doc.access_level or "Internal",
        payload.text,
    )

    return {"status": "Success", "message": "Document text updated. Re-scanning."}


@app.post("/api/admin/import")
def import_missing_files(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin),
):
    # Only load the columns we actually need — skip blob columns
    existing_docs = (
        db.query(DocumentMetadata)
        .options(
            load_only(
                DocumentMetadata.id,
                DocumentMetadata.filename,
                DocumentMetadata.risk_description,
                DocumentMetadata.access_level,
            )
        )
        .all()
    )
    existing_filenames = {d.filename for d in existing_docs}

    try:
        # Query distinct source filenames directly from PGVector's metadata table.
        # This avoids loading thousands of embedding rows into Python memory.
        from database import engine as db_engine
        from sqlalchemy import text as sql_text

        with db_engine.connect() as conn:
            result = conn.execute(
                sql_text(
                    "SELECT DISTINCT cmetadata->>'source' "
                    "FROM langchain_pg_embedding "
                    "WHERE collection_id = ("
                    "  SELECT uuid FROM langchain_pg_collection "
                    "  WHERE name = 'ccl_docintel_vectors'"
                    ")"
                )
            )
            synced_sources = {row[0] for row in result if row[0]}
    except Exception:
        synced_sources = set()

    files_on_disk = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
    imported = 0

    # Rescan files that are in Postgres but failed scanning previously
    for doc in existing_docs:
        if "Scanner failed" in (doc.risk_description or ""):
            doc.risk_level = "Scanning..."
            doc.risk_description = "AI compliance scan in progress."
            db.commit()
            background_tasks.add_task(
                _run_ai_background, doc.id, doc.filename, doc.access_level or "Internal"
            )
            imported += 1

    for filename in files_on_disk:
        if filename not in existing_filenames or filename not in synced_sources:
            file_path = os.path.join(UPLOAD_DIR, filename)
            parsed_text = extract_text_from_file(file_path)
            char_count = len(parsed_text)

            clean_text = re.sub(
                r"\[Image Metadata\].*?(?=\n\[|$)", "", parsed_text, flags=re.DOTALL
            )
            clean_text = re.sub(
                r"\[EXIF Data\].*?(?=\n\[|$)", "", clean_text, flags=re.DOTALL
            )
            clean_text = re.sub(
                r"\[OCR Error\].*?(?=\n\[|$)", "", clean_text, flags=re.DOTALL
            )
            clean_text = clean_text.strip().lower()

            if clean_text:
                hash_material = clean_text.encode("utf-8")
            else:
                try:
                    from PIL import Image as PilImage

                    img = (
                        PilImage.open(file_path)
                        .convert("RGB")
                        .resize((32, 32), PilImage.Resampling.LANCZOS)
                    )
                    hash_material = img.tobytes()
                except Exception:
                    with open(file_path, "rb") as f:
                        hash_material = f.read()
            file_hash = hashlib.sha256(hash_material).hexdigest()

            if filename not in existing_filenames:
                db_record = DocumentMetadata(
                    filename=filename,
                    content_type="application/octet-stream",
                    char_count=char_count,
                    extracted_text=parsed_text[:50000],
                    risk_level="Scanning...",
                    risk_description="AI compliance scan in progress.",
                    content_hash=file_hash,
                )
                db.add(db_record)
                db.commit()
                doc_id = db_record.id
            else:
                doc = (
                    db.query(DocumentMetadata)
                    .filter(DocumentMetadata.filename == filename)
                    .first()
                )
                doc.risk_level = "Scanning..."
                doc.risk_description = "AI compliance scan in progress."
                doc.extracted_text = parsed_text[:50000]
                db.commit()
                doc_id = doc.id

            background_tasks.add_task(_run_ai_background, doc_id, filename)
            imported += 1

    return {
        "status": "Success",
        "imported": imported,
        "message": "Missing files are being imported in the background.",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
