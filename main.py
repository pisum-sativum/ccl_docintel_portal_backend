import time
_boot_start = time.time()
print(f"[BOOT] main.py import started at {_boot_start}")

import os
import hashlib
import re
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

# Import custom application modules
from extractor import extract_text_from_file
from database import init_db, get_db, DocumentMetadata, User, SessionLocal
from rag_engine import inject_text_into_vector_store, query_document_intelligence, scan_text_for_compliance_risks, extract_document_metadata
from auth import (
    verify_password, create_access_token,
    get_current_user, require_admin
)

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
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled. Contact your administrator.")
    token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role, "username": user.username}

@app.get("/api/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Returns the currently authenticated user's info."""
    return current_user

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "system": "CCL DocIntel Core Engine",
        "database": "Neon Cloud PostgreSQL Connected"
    }

@app.post("/api/chat")
def handle_chat(payload: ChatQuery, current_user: dict = Depends(get_current_user)):
    """RAG chat endpoint — open to all authenticated users."""
    ai_response_text = query_document_intelligence(payload.query_text, payload.history, user_role=current_user["role"])
    return {"sender": "bot", "text": ai_response_text}

@app.get("/api/analytics/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    total_docs = db.query(func.count(DocumentMetadata.id)).scalar() or 0
    total_chars = db.query(func.sum(DocumentMetadata.char_count)).scalar() or 0
    vectorized_nodes = int(total_chars / 200) if total_chars > 0 else 0
    active_flags = db.query(func.count(DocumentMetadata.id)).filter(
        DocumentMetadata.risk_level.in_(["High", "Medium"])
    ).scalar() or 0
    if total_docs > 0:
        integrity_score = round((1 - (active_flags / total_docs)) * 100, 1)
    else:
        integrity_score = 100.0
    return {
        "vectorizedNodes": vectorized_nodes,
        "activeFlags": active_flags,
        "complianceIntegrity": f"{integrity_score}%"
    }

@app.get("/api/compliance/alerts")
def get_compliance_alerts(db: Session = Depends(get_db)):
    """Fetches AI-flagged compliance violations."""
    try:
        flagged_docs = db.query(DocumentMetadata).filter(
            DocumentMetadata.risk_level.in_(["High", "Medium"])
        ).order_by(DocumentMetadata.upload_date.desc()).all()
        alerts = []
        for doc in flagged_docs:
            alerts.append({
                "id": doc.id,
                "doc": doc.filename,
                "type": f"{doc.risk_level} Risk Rule Violation",
                "risk": doc.risk_level,
                "desc": doc.risk_description
            })
        if not alerts:
            return [{
                "id": 0,
                "doc": "System Stable",
                "type": "Operational Stability",
                "risk": "None",
                "desc": "No active vulnerabilities flagged across repository nodes."
            }]
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
@app.post("/api/documents")
def list_uploaded_documents(skip: int = 0, limit: int = 5, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    Returns a paginated slice of documents.
    NOTE: Auto-pruning has been REMOVED because Render uses an ephemeral filesystem.
    Physical files are lost on every restart, but extracted_text lives in the cloud
    PostgreSQL DB and must be preserved. Delete via the DELETE endpoint instead.
    """
    base_query = db.query(DocumentMetadata)
    if current_user["role"] != "admin":
        base_query = base_query.filter(DocumentMetadata.access_level.in_(["Public", "Internal"]))
        
    total_count = base_query.count()
    documents = base_query.order_by(
        DocumentMetadata.upload_date.desc()
    ).offset(skip).limit(limit).all()

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
        ]
    }

# ─────────────────────────────────────────────────────────────────────────────
# ADMIN-ONLY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_ai_background(doc_id: int, parsed_text: str, filename: str, access_level: str = "Internal"):
    """
    Runs the slow AI compliance scan and vector indexing in a background thread.
    Updates the DB record when done so the dashboard reflects the final result.
    """
    try:
        # 1. AI compliance scan (Gemini API call, ~2-5 seconds)
        audit_result = scan_text_for_compliance_risks(parsed_text, filename)
        
        # 1.5 Extract metadata
        metadata_result = extract_document_metadata(parsed_text, filename)

        # 2. Vector store injection (CPU embedding, ~1-3 seconds)
        inject_text_into_vector_store(parsed_text, filename, access_level)

        # 3. Update the DB record with the final AI results
        db = SessionLocal()
        try:
            doc = db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
            if doc:
                doc.risk_level = audit_result["risk_level"]
                doc.risk_description = audit_result["description"]
                doc.department = metadata_result.get("department", "Unknown")
                doc.doc_type = metadata_result.get("doc_type", "Unknown")
                doc.summary = metadata_result.get("summary", "")
                db.commit()
                print(f"[BG] AI scan done for '{filename}': {audit_result['risk_level']}")
        finally:
            db.close()
    except Exception as e:
        print(f"[BG] Background AI task failed for '{filename}': {e}")


@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    force: bool = Form(False),
    access_level: str = Form("Internal"),
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin)  # 🔒 ADMIN ONLY
):
    try:
        content = await file.read()

        # ── 1. Check for filename conflicts ──
        existing_by_name = db.query(DocumentMetadata).filter(
            DocumentMetadata.filename == file.filename
        ).first()
        if existing_by_name:
            raise HTTPException(
                status_code=409,
                detail=f"A file with the name '{file.filename}' already exists in the system."
            )

        # ── 2. Save to disk ──
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # ── 3. Extract text (fast local operation) ──
        parsed_text = extract_text_from_file(file_path)
        char_count = len(parsed_text)

        # ── 4. Content-aware duplicate detection (Fuzzy/Normalized) ──
        clean_text = re.sub(r'\[Image Metadata\].*?(?=\n\[|$)', '', parsed_text, flags=re.DOTALL)
        clean_text = re.sub(r'\[EXIF Data\].*?(?=\n\[|$)', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'\[OCR Error\].*?(?=\n\[|$)', '', clean_text, flags=re.DOTALL)
        clean_text = clean_text.strip().lower()

        existing_by_hash = None
        if clean_text:
            normalized = re.sub(r'\W+', '', clean_text)
            hash_material = normalized.encode('utf-8')
            file_hash = hashlib.sha256(hash_material).hexdigest()
            
            incoming_words = set(clean_text.split())
            all_existing = db.query(DocumentMetadata).all()
            for doc in all_existing:
                if not doc.extracted_text: continue
                existing_clean = re.sub(r'\[Image Metadata\].*?(?=\n\[|$)', '', doc.extracted_text, flags=re.DOTALL)
                existing_clean = re.sub(r'\[EXIF Data\].*?(?=\n\[|$)', '', existing_clean, flags=re.DOTALL)
                existing_clean = existing_clean.strip().lower()
                existing_words = set(existing_clean.split())
                
                intersection = incoming_words.intersection(existing_words)
                union = incoming_words.union(existing_words)
                if union:
                    similarity = len(intersection) / len(union)
                    if similarity > 0.85:  # 85% overlap is considered a duplicate
                        existing_by_hash = doc
                        break
        else:
            try:
                from PIL import Image as PilImage
                img = PilImage.open(file_path).convert("RGB").resize((32, 32), PilImage.Resampling.LANCZOS)
                hash_material = img.tobytes()
            except Exception:
                hash_material = content

            file_hash = hashlib.sha256(hash_material).hexdigest()
            existing_by_hash = db.query(DocumentMetadata).filter(
                DocumentMetadata.content_hash == file_hash
            ).first()
        
        if existing_by_hash:
            if not force:
                os.remove(file_path)
                raise HTTPException(
                    status_code=409,
                    detail={
                        "type": "similar_content",
                        "existing_filename": existing_by_hash.filename,
                        "message": f"This file contains the same data as '{existing_by_hash.filename}'."
                    }
                )
            # If force=True, we proceed and keep the file despite the similarity.

        # ── 5. Save a "pending" DB record immediately ──
        db_record = DocumentMetadata(
            filename=file.filename,
            content_type=file.content_type,
            char_count=char_count,
            extracted_text=parsed_text[:50000],   # Store up to 50k chars for viewing
            risk_level="Scanning...",
            risk_description="AI compliance scan in progress.",
            content_hash=file_hash,
            access_level=access_level,
            raw_file_data=content
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        # ── 6. Offload slow AI work to background thread ──
        background_tasks.add_task(_run_ai_background, db_record.id, parsed_text, file.filename, access_level)

        # ── 7. Return immediately ──
        return {
            "status": "Success",
            "filename": db_record.filename,
            "characterCount": char_count,
            "previewText": parsed_text[:2000],
            "message": "File saved. AI compliance scan running in background."
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/documents/{doc_id}/text")
@app.post("/api/documents/{doc_id}/text")
def get_document_text(doc_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Returns the stored extracted text for a document (for the View modal)."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if current_user["role"] != "admin" and doc.access_level == "Confidential":
        raise HTTPException(status_code=403, detail="Access denied. Confidential document.")
    return {
        "id": doc.id,
        "filename": doc.filename,
        "char_count": doc.char_count,
        "risk_level": doc.risk_level,
        "risk_description": doc.risk_description,
        "extracted_text": doc.extracted_text or "No text content available for this file.",
    }

@app.get("/api/documents/{doc_id}/file")
@app.post("/api/documents/{doc_id}/file")
def get_document_file(doc_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Returns the physical file for native viewing (PDF/Images)."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if current_user["role"] != "admin" and doc.access_level == "Confidential":
        raise HTTPException(status_code=403, detail="Access denied. Confidential document.")
    # Return from database if available (this survives Render restarts)
    if getattr(doc, "raw_file_data", None) is not None:
        headers = {}
        if doc.content_type == "application/pdf":
            headers["Content-Disposition"] = f'inline; filename="{doc.filename}"'
        return Response(content=doc.raw_file_data, media_type=doc.content_type, headers=headers)
        
    # Fallback to local disk (might be wiped by Render, but keep just in case)
    file_path = os.path.join(UPLOAD_DIR, doc.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Physical file missing from disk and no backup in database.")
    
    headers = {}
    if doc.content_type == "application/pdf":
        headers["Content-Disposition"] = f'inline; filename="{doc.filename}"'
        
    return FileResponse(file_path, media_type=doc.content_type, filename=doc.filename, headers=headers)

@app.delete("/api/documents/{doc_id}")
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin)  # 🔒 ADMIN ONLY
):
    """Deletes a document record from DB, disk, and vector store."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # 1. Remove from vector store
    try:
        from rag_engine import delete_document_from_vector_store
        delete_document_from_vector_store(doc.filename)
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
    _admin: dict = Depends(require_admin)
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
    hash_material = payload.text.encode('utf-8')
    doc.content_hash = hashlib.sha256(hash_material).hexdigest()
    
    db.commit()
    
    # 2. Re-run AI indexing background task
    background_tasks.add_task(_run_ai_background, doc.id, payload.text, doc.filename)
    
    return {"status": "Success", "message": "Document text updated. Re-scanning."}

@app.post("/api/admin/import")
def import_missing_files(background_tasks: BackgroundTasks, db: Session = Depends(get_db), _admin: dict = Depends(require_admin)):
    existing_docs = db.query(DocumentMetadata).all()
    existing_filenames = {d.filename for d in existing_docs}
    
    try:
        from rag_engine import get_vector_db
        # Get all synced sources from PGVector using SQL filter
        vdb = get_vector_db()
        results = vdb.similarity_search(".", k=10000)  # fetch all
        synced_sources = set(d.metadata.get("source") for d in results if d.metadata.get("source"))
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
            background_tasks.add_task(_run_ai_background, doc.id, doc.extracted_text or "", doc.filename)
            imported += 1

    for filename in files_on_disk:
        if filename not in existing_filenames or filename not in synced_sources:
            file_path = os.path.join(UPLOAD_DIR, filename)
            parsed_text = extract_text_from_file(file_path)
            char_count = len(parsed_text)
            
            clean_text = re.sub(r'\[Image Metadata\].*?(?=\n\[|$)', '', parsed_text, flags=re.DOTALL)
            clean_text = re.sub(r'\[EXIF Data\].*?(?=\n\[|$)', '', clean_text, flags=re.DOTALL)
            clean_text = re.sub(r'\[OCR Error\].*?(?=\n\[|$)', '', clean_text, flags=re.DOTALL)
            clean_text = clean_text.strip().lower()
            
            if clean_text:
                hash_material = clean_text.encode('utf-8')
            else:
                try:
                    from PIL import Image as PilImage
                    img = PilImage.open(file_path).convert("RGB").resize((32, 32), PilImage.Resampling.LANCZOS)
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
                    content_hash=file_hash
                )
                db.add(db_record)
                db.commit()
                doc_id = db_record.id
            else:
                doc = db.query(DocumentMetadata).filter(DocumentMetadata.filename == filename).first()
                doc.risk_level = "Scanning..."
                doc.risk_description = "AI compliance scan in progress."
                doc.extracted_text = parsed_text[:50000]
                db.commit()
                doc_id = doc.id
            
            background_tasks.add_task(_run_ai_background, doc_id, parsed_text, filename)
            imported += 1
            
    return {"status": "Success", "imported": imported, "message": "Missing files are being imported in the background."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)