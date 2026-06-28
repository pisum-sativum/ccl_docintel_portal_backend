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
        _embedding_engine = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=os.getenv("GEMINI_API_KEY", "")
        )
    return _embedding_engine

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        from langchain_community.vectorstores import Chroma
        _vector_db = Chroma(
            persist_directory="./chroma_db_v3",
            embedding_function=get_embedding_engine()
        )
    return _vector_db

def get_llm():
    """Lazy-initialize the LLM — import AND instantiate only on first call.
    This prevents blocking Uvicorn port binding on Render."""
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
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
def inject_text_into_vector_store(raw_text: str, filename: str, access_level: str = "Internal") -> bool:
    """
    Indexes document text into ChromaDB.
    FIX: Deletes any previously stored chunks for this filename first,
    preventing duplicate context build-up on re-uploads.
    """
    if not raw_text or raw_text.startswith("[Parsing Failure]"):
        return False
    try:
        # ── Delete old chunks for this file before re-indexing ────────────
        try:
            existing = get_vector_db()._collection.get(where={"source": filename})
            if existing and existing.get("ids"):
                get_vector_db()._collection.delete(ids=existing["ids"])
                print(f"🗑️  [DEDUP] Removed {len(existing['ids'])} stale chunks for '{filename}'")
        except Exception as del_err:
            print(f"⚠️  [DEDUP WARN] Could not remove old chunks: {del_err}")

        # ── Split and re-index ────────────────────────────────────────────
        chunks = get_text_splitter().split_text(raw_text)
        metadata_tags = [{"source": filename, "access_level": access_level} for _ in chunks]
        get_vector_db().add_texts(texts=chunks, metadatas=metadata_tags)
        print(f"📦 [VECTOR INDEXED] Committed {len(chunks)} chunks for '{filename}'")
        return True
    except Exception as e:
        print(f"❌ [VECTOR CRASH] Indexing aborted: {str(e)}")
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
    Fetches ALL stored chunks directly from ChromaDB (bypasses semantic scoring).
    Returns list of objects with .page_content and .metadata attributes.
    """
    try:
        kwargs = {"include": ["documents", "metadatas"]}
        if filter_criteria:
            kwargs["where"] = filter_criteria
        result = get_vector_db()._collection.get(**kwargs)
        docs  = result.get("documents", [])
        metas = result.get("metadatas", []) or [{}] * len(docs)

        class _Chunk:
            def __init__(self, text, meta):
                self.page_content = text
                self.metadata     = meta

        return [_Chunk(d, m) for d, m in zip(docs, metas)]
    except Exception as e:
        print(f"⚠️ [FULL SCAN ERROR] {e}")
        return []


def _keyword_scan_all(query: str, top_n: int = 6, filter_criteria=None) -> list:
    """
    Case-insensitive keyword scan across ALL ChromaDB chunks.
    Skipped automatically if the collection is very large (>3000 chunks)
    to avoid linear-scan performance degradation at scale.
    """
    # ── Scale guard: skip full scan for very large DBs ────────────────────
    try:
        total_chunks = get_vector_db()._collection.count()
        if total_chunks > 3000:
            print(f"⚡ [SCALE GUARD] {total_chunks} chunks — skipping full keyword scan")
            return []
    except Exception:
        pass

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

        if not merged:
            return "I couldn't find any relevant text inside the uploaded compliance database to answer that question."

        # ── Step D: Build context with source labels ──────────────────────
        context_parts = []
        for doc in merged[:12]:
            src = doc.metadata.get("source", "Unknown Document")
            context_parts.append(f"[Source: {src}]\n{doc.page_content}")
        context_text = "\n\n---\n\n".join(context_parts)

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
            "Answer the user's question using ONLY the provided document context below.\n\n"
            "IMPORTANT RULES:\n"
            "- Numbers may appear formatted differently: '50000', 'Rs. 50,000', '50,000/-', '₹50,000'. Treat them as equivalent.\n"
            "- Acronyms and proper nouns may differ in case (e.g. 'gssoc' matches 'GSSoC'). Match case-insensitively.\n"
            "- ALWAYS cite the source document in your answer, e.g.: [Source: fee_letter.pdf]\n"
            "- Use the conversation history (if provided) to understand follow-up questions.\n"
            "- If the answer truly cannot be found in the context, say: 'I cannot find that information in the current database.'\n"
            "- Do NOT make up facts.\n"
            f"{history_section}\n"
            f"--- DOCUMENT CONTEXT ---\n{context_text}\n\n"
            f"User Question: {user_question}"
        )

        # ── Step G: Invoke Gemini ─────────────────────────────────────────
        response = get_llm().invoke(system_instruction)
        return response.content

    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
            return "⚠️ The AI service has temporarily reached its request limit. Please wait about a minute and try your question again."
        return "⚠️ We encountered a temporary issue connecting to the AI engine. Please try again."


# ── 5. Compliance Risk Scanner ────────────────────────────────────────────────
def scan_text_for_compliance_risks(extracted_text: str, filename: str = "Unknown") -> dict:
    """
    Uses the Gemini LLM to scan extracted document text and filename for compliance risks.
    Returns a dict with 'risk_level' (High/Medium/None) and 'description'.
    """
    if not extracted_text or extracted_text.startswith("[Parsing"):
        return {"risk_level": "None", "description": "Unparsable file text layout."}

    try:
        # We instruct the AI to act as a strict inspector
        compliance_prompt = (
            f"You are an industrial compliance auditor. Analyze the following document text "
            f"and its filename ('{filename}') for any operational hazards, safety violations, "
            f"missing safety protocols, financial discrepancies, expired certifications, "
            f"or any errors, inconsistencies, or suspicious patterns in the filename itself.\n\n"
            f"Respond in exactly this format:\n"
            f"RISK: [High, Medium, or None]\n"
            f"REASON: [A short 1-sentence description of the hazard found]\n\n"
            f"--- DOCUMENT TEXT ---\n{extracted_text[:4000]}"  # Scan the first 4000 characters
        )

        response = get_llm().invoke(compliance_prompt)
        # Strip all markdown bold/italic characters to ensure clean matching
        response_text = response.content.lower().replace("*", "").replace("`", "").replace('"', '').replace("'", "")

        # Parse the AI response fields
        if "risk: high" in response_text or "risk level: high" in response_text:
            risk = "High"
        elif "risk: medium" in response_text or "risk level: medium" in response_text:
            risk = "Medium"
        else:
            risk = "None"

        reason = "All standard compliance check metrics validated."
        # Use regex to robustly find the reason part, ignoring markdown
        import re
        match = re.search(r'reason:\s*(.*)', response_text, re.IGNORECASE)
        if match:
            reason = match.group(1).strip()

        return {"risk_level": risk, "description": reason}
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            return {"risk_level": "None", "description": "AI scan delayed due to service rate limits."}
        return {"risk_level": "None", "description": "Scanner failed to process document."}

def extract_document_metadata(text: str, filename: str) -> dict:
    """
    Uses Gemini LLM to extract Department, Document Type, and a Summary.
    """
    if not text:
        return {"department": "Unknown", "doc_type": "Unknown", "summary": "No text extracted."}
        
    prompt = (
        f"You are a document metadata extraction engine. Read the following document ('{filename}') "
        f"and extract three things:\n"
        f"1. Department: (e.g. HR, Engineering, Legal, Safety, Finance, General)\n"
        f"2. Type: (e.g. Form, Report, Guidelines, Policy, Memo, Unknown)\n"
        f"3. Summary: (A concise 1-2 sentence summary of the document)\n\n"
        f"Respond ONLY in this exact format:\n"
        f"DEPARTMENT: [department]\n"
        f"TYPE: [type]\n"
        f"SUMMARY: [summary]\n\n"
        f"--- DOCUMENT TEXT ---\n{text[:4000]}"
    )
    
    try:
        response = get_llm().invoke(prompt)
        content = response.content.replace("*", "").replace("`", "")
        dept = "Unknown"
        doc_type = "Unknown"
        summary = "No summary available."
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("DEPARTMENT:"):
                dept = line.replace("DEPARTMENT:", "").strip()
            elif line.startswith("TYPE:"):
                doc_type = line.replace("TYPE:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
                
        return {"department": dept, "doc_type": doc_type, "summary": summary}
    except Exception:
        return {"department": "Unknown", "doc_type": "Unknown", "summary": "Extraction failed due to an error."}