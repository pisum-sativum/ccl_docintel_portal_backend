# ── Lightweight stdlib-only imports at module level ──────────────────────────
# ALL heavy langchain/sentence-transformer imports are deferred to the first
# time they are actually needed.  This lets Uvicorn bind its port in < 1 s.
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ── 1. Initialize Engines ─────────────────────────────────────────────────────
_embedding_engine = None
_vector_db = None
_llm = None
_text_splitter = None

def get_embedding_engine():
    global _embedding_engine
    if _embedding_engine is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        class PatchedEmbeddings(GoogleGenerativeAIEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                # Fix LangChain list index out of range bug for batch embeddings
                return [self.embed_query(t) for t in texts]

        _embedding_engine = PatchedEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY", "")
        )
    return _embedding_engine

def get_vector_db():
    """
    Returns a persistent PGVector store backed by Neon PostgreSQL.
    Survives Render restarts — no data loss.
    """
    global _vector_db
    if _vector_db is None:
        from langchain_postgres.vectorstores import PGVector
        connection_string = os.getenv("DATABASE_URL", "")
        # langchain-postgres requires postgresql:// scheme (not postgres://)
        if connection_string.startswith("postgres://"):
            connection_string = connection_string.replace("postgres://", "postgresql+psycopg://", 1)
        elif connection_string.startswith("postgresql://"):
            connection_string = connection_string.replace("postgresql://", "postgresql+psycopg://", 1)
        _vector_db = PGVector(
            embeddings=get_embedding_engine(),
            collection_name="ccl_docintel_vectors",
            connection=connection_string,
            use_jsonb=True,
        )
    return _vector_db

def get_llm():
    """Lazy-initialize the LLM — import AND instantiate only on first call.
    This prevents blocking Uvicorn port binding on Render."""
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY", ""),
            temperature=0.2,
        )
    return _llm

def get_text_splitter():
    """Lazy-initialize the text splitter to avoid importing langchain at module load."""
    global _text_splitter
    if _text_splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
            length_function=len
        )
    return _text_splitter

def warm_up():
    get_vector_db()


# ── 2. Data Ingestion ─────────────────────────────────────────────────────────
def delete_document_from_vector_store(filename: str):
    """
    Removes all vector chunks for a given filename from PGVector.
    """
    try:
        vdb = get_vector_db()
        vdb.delete(ids=None, filter={"source": filename})
        print(f"[DEDUP] Removed stale chunks for '{filename}'")
    except Exception as del_err:
        print(f"[DEDUP WARN] Could not remove old chunks: {del_err}")


def inject_text_into_vector_store(raw_text: str, filename: str, access_level: str = "Internal") -> bool:
    """
    Indexes document text into PGVector (persistent Neon PostgreSQL).
    Deletes any previously stored chunks for this filename first,
    preventing duplicate context build-up on re-uploads.
    """
    if not raw_text or raw_text.startswith("[Parsing Failure]"):
        return False
    try:
        # ── Delete old chunks for this file before re-indexing ────────────
        delete_document_from_vector_store(filename)

        # ── Split and re-index ────────────────────────────────────────────
        chunks = get_text_splitter().split_text(raw_text)
        if not chunks:
            return False
        metadata_tags = [{"source": filename, "access_level": access_level} for _ in chunks]
        get_vector_db().add_texts(texts=chunks, metadatas=metadata_tags)
        print(f"[VECTOR INDEXED] Committed {len(chunks)} chunks for '{filename}'")
        return True
    except Exception as e:
        print(f"[VECTOR CRASH] Indexing aborted: {str(e)}")
        return False


# ── 3. Helpers ────────────────────────────────────────────────────────────────
def _normalize_number_query(query: str) -> list[str]:
    """
    Generate number-format variants for a query containing digits.
    e.g. '50000' → ['50,000', 'Rs. 50,000', 'Rs. 50,000 /-', '₹50,000', ...]
    """
    raw_numbers = re.findall(r'\d+', query.replace(',', ''))
    variants = []
    for num_str in raw_numbers:
        n = int(num_str)
        if n >= 1000:
            formatted = f"{n:,}"
            variants += [
                num_str,
                formatted,
                f"Rs. {formatted}",
                f"Rs {num_str}",
                f"Rs. {formatted} /-",
                f"Rs. {formatted}/-",
                f"₹{formatted}",
                f"₹ {formatted}",
            ]
        else:
            variants.append(num_str)
    return list(set(variants))


def _get_all_chunks_from_db(filter_criteria=None) -> list:
    """
    Fetches ALL stored chunks from PGVector using a broad similarity search
    (bypasses semantic scoring for keyword matching).
    Returns list of langchain Document objects with .page_content and .metadata.
    """
    try:
        vdb = get_vector_db()
        # Use a max-fetch similarity search with a neutral query
        kwargs = {"k": 3000}
        if filter_criteria:
            # Convert Chroma-style filter to PGVector filter format
            access_filter = filter_criteria.get("access_level", {}).get("$in")
            if access_filter:
                kwargs["filter"] = {"access_level": {"$in": access_filter}}
        results = vdb.similarity_search("compliance document text", **kwargs)
        return results
    except Exception as e:
        print(f"[FULL SCAN ERROR] {e}")
        return []


def _keyword_scan_all(query: str, top_n: int = 6, filter_criteria=None) -> list:
    """
    Case-insensitive keyword scan across all PGVector chunks.
    Uses the fetched chunk list from _get_all_chunks_from_db.
    """
    all_chunks = _get_all_chunks_from_db(filter_criteria)
    if not all_chunks:
        return []

    raw_tokens     = set(query.lower().split())
    number_variants = set(v.lower() for v in _normalize_number_query(query))
    tokens         = raw_tokens | number_variants

    scored = []
    for chunk in all_chunks:
        text_lower = chunk.page_content.lower()
        score = sum(1 for t in tokens if t and t in text_lower)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_n]]


# ── 4. Core Query Pipeline ────────────────────────────────────────────────────
def query_document_intelligence(user_question: str, history: list[dict] = None, user_role: str = "viewer") -> str:
    """
    Hybrid retrieval RAG pipeline with:
      - Semantic search (k=8) + full keyword scan for exact term matching
      - Source attribution: Gemini is instructed to cite document filenames
      - Conversation history: last 4 exchanges passed as context
      - Scale guard: keyword scan disabled for very large collections
      - RBAC logic based on user_role
    """
    if history is None:
        history = []

    try:
        # ── RBAC Filter ──
        search_filter = None
        if user_role != "admin":
            search_filter = {"access_level": {"$in": ["Public", "Internal"]}}

        # ── Step A: Semantic search ───────────────────────────────────────
        semantic_docs = get_vector_db().similarity_search(user_question, k=8, filter=search_filter)

        # ── Step B: Full keyword scan (always runs, scale-guarded) ────────
        keyword_docs = _keyword_scan_all(user_question, top_n=6, filter_criteria=search_filter)

        # ── Step C: Merge — keyword results first (higher precision) ──────
        seen, merged = set(), []
        for doc in (keyword_docs + semantic_docs):
            key = doc.page_content[:120].strip()
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        # ── Step D: Build context with source labels ──────────────────────
        context_parts = []
        for doc in merged[:12]:
            src = doc.metadata.get("source", "Unknown Document")
            context_parts.append(f"[Source: {src}]\n{doc.page_content}")
        context_text = "\n\n---\n\n".join(context_parts)
        if not context_text:
            context_text = "[NO RELEVANT DOCUMENTS FOUND IN DATABASE. PLEASE PROVIDE A GENERAL ANSWER BASED ON YOUR KNOWLEDGE, BUT WARN THE USER NO SPECIFIC COMPLIANCE DOCUMENTS WERE FOUND.]"

        # ── Step E: Build conversation history string ─────────────────────
        history_text = ""
        if history:
            recent = history[-6:]  # last 3 exchanges (user + bot = 2 msgs each)
            lines = []
            for msg in recent:
                role = "User" if msg.get("sender") == "user" else "Assistant"
                lines.append(f"{role}: {msg.get('text', '')}")
            history_text = "\n".join(lines)

        # ── Step F: Construct system prompt ──────────────────────────────
        history_section = (
            f"\n--- CONVERSATION HISTORY ---\n{history_text}\n"
            if history_text else ""
        )

        system_instruction = (
            "You are CCL DocIntel, an expert corporate AI compliance assistant.\n"
            "Answer the user's question using the provided document context below.\n"
            "If no context is provided, you may answer using your general knowledge but MUST clearly state that you couldn't find specific uploaded documents.\n\n"
            "IMPORTANT RULES:\n"
            "- Numbers may appear formatted differently: '50000', 'Rs. 50,000', '50,000/-', '₹50,000'. Treat them as equivalent.\n"
            "- Acronyms and proper nouns may differ in case (e.g. 'gssoc' matches 'GSSoC'). Match case-insensitively.\n"
            "- If the answer truly cannot be found in the context, say: 'I cannot find that information in the current database.'\n"
            "- Do NOT make up facts.\n"
            f"{history_section}\n"
            f"--- DOCUMENT CONTEXT ---\n{context_text}\n\n"
            f"User Question: {user_question}"
        )

        # ── Step G: Invoke Gemini with automatic retries for rate limits ──
        import time
        max_attempts = 2  # Fail fast
        for attempt in range(max_attempts):
            try:
                response = get_llm().invoke(system_instruction)
                return response.content
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    if attempt < max_attempts - 1:
                        time.sleep(5)  # Short sleep
                        continue
                    return "⚠️ The AI service has temporarily reached its request limit. Please wait about a minute and try your question again."
                return f"⚠️ We encountered a temporary issue connecting to the AI engine. ERROR: {err_str}"
                
    except Exception as e:
        return f"⚠️ We encountered a temporary issue connecting to the AI engine. ERROR: {str(e)}"


def analyze_document(extracted_text: str, filename: str) -> dict:
    """
    Combines compliance scanning and metadata extraction into a single Gemini LLM call.
    This cuts latency in half and prevents API rate limit exhaustion.
    """
    if not extracted_text:
        return {
            "risk_level": "None", 
            "description": "No text extracted.", 
            "department": "Unknown", 
            "doc_type": "Unknown", 
            "summary": "No text extracted."
        }

    try:
        prompt = (
            f"You are a strict cybersecurity and document analysis AI agent. Review the following text "
            f"and its filename ('{filename}') for extreme operational hazards, critical safety violations, "
            f"exposed credentials, and security risks. Perform both a compliance scan and metadata extraction.\n\n"
            f"**COMPLIANCE SCAN INSTRUCTIONS:**\n"
            f"Be lenient on normal business files! Do NOT flag minor administrative issues, normal guidelines, or benign text as High/Medium risk.\n"
            f"HOWEVER, if the document contains passwords, hacking instructions, malicious payloads, or if its filename indicates it is a security issue, "
            f"you MUST mark it as 'High' risk.\n\n"
            f"**METADATA INSTRUCTIONS:**\n"
            f"Extract the Department (HR, Legal, IT, etc), Type (Form, Policy, Memo, etc), and a concise 1-2 sentence Summary.\n\n"
            f"Respond ONLY in this EXACT format (do not add markdown blocks):\n"
            f"RISK: [High, Medium, or None]\n"
            f"REASON: [A short 1-sentence description of the hazard, or 'All clear']\n"
            f"DEPARTMENT: [department]\n"
            f"TYPE: [type]\n"
            f"SUMMARY: [summary]\n\n"
            f"--- DOCUMENT TEXT ---\n{extracted_text[:4000]}"
        )

        import time
        max_attempts = 2  # Fail fast to avoid hanging the queue
        for attempt in range(max_attempts):
            try:
                response = get_llm().invoke(prompt)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    if attempt < max_attempts - 1:
                        # Short sleep to recover from minor rate bumps, but don't hold the lock forever
                        time.sleep(5)
                        continue
                    return {
                        "risk_level": "Error", 
                        "description": f"AI scan aborted due to Gemini API limits. Please wait a minute and retry. (Details: {err_str})",
                        "department": "Unknown", 
                        "doc_type": "Unknown", 
                        "summary": "Scan failed (API limits)."
                    }
                return {
                    "risk_level": "Error", 
                    "description": f"Scanner failed: {err_str}",
                    "department": "Unknown", 
                    "doc_type": "Unknown", 
                    "summary": "Extraction failed due to an error."
                }

        content = response.content.replace("*", "").replace("`", "")
        
        result = {
            "risk_level": "None",
            "description": "All clear.",
            "department": "Unknown",
            "doc_type": "Unknown",
            "summary": "No summary available."
        }
        
        for line in content.split('\n'):
            line = line.strip()
            if line.upper().startswith("RISK:"):
                risk = line[5:].strip().title()
                if "High" in risk: result["risk_level"] = "High"
                elif "Medium" in risk: result["risk_level"] = "Medium"
            elif line.upper().startswith("REASON:"):
                result["description"] = line[7:].strip()
            elif line.upper().startswith("DEPARTMENT:"):
                result["department"] = line[11:].strip()
            elif line.upper().startswith("TYPE:"):
                result["doc_type"] = line[5:].strip()
            elif line.upper().startswith("SUMMARY:"):
                result["summary"] = line[8:].strip()
                
        return result

    except Exception as e:
        err_str = str(e)
        return {
            "risk_level": "Error", 
            "description": f"Scanner failed: {err_str}",
            "department": "Unknown", 
            "doc_type": "Unknown", 
            "summary": "Extraction failed due to an error."
        }