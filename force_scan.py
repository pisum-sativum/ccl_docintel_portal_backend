import time
from database import SessionLocal, DocumentMetadata
from rag_engine import scan_text_for_compliance_risks

db = SessionLocal()
docs = db.query(DocumentMetadata).filter(DocumentMetadata.filename.in_(['audit.txt', 'secure.txt', 'Audit_Discrepancy_Log.txt..txt'])).all()
texts = [(d.id, d.extracted_text, d.filename) for d in docs]
db.close()

for doc_id, text, filename in texts:
    while True:
        res = scan_text_for_compliance_risks(text, filename)
        desc = res['description'].lower()
        if 'rate limit' not in desc and 'failed' not in desc:
            db2 = SessionLocal()
            d = db2.query(DocumentMetadata).get(doc_id)
            d.risk_level = res['risk_level']
            d.risk_description = res['description']
            db2.commit()
            db2.close()
            print(f"SUCCESS {filename}: {res['risk_level']}", flush=True)
            break
        else:
            print(f"Rate limit hit for {filename}. Retrying in 10s...", flush=True)
            time.sleep(10)
