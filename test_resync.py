import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv(r'c:\Users\KIIT0001\Desktop\CCL-Project\ccl-docintel-backend\.env')

sys.path.append(r'c:\Users\KIIT0001\Desktop\CCL-Project\ccl-docintel-backend')

from database import SessionLocal, DocumentMetadata
from main import _run_ai_background

db = SessionLocal()
existing_docs = db.query(DocumentMetadata).all()

for doc in existing_docs:
    if doc.risk_description and "Scanner failed" in doc.risk_description:
        print(f"Rescanning: {doc.filename}")
        try:
            # We must run it synchronously here to see if it works
            _run_ai_background(doc.id, doc.extracted_text or "", doc.filename)
            print(f"Success for {doc.filename}")
        except Exception as e:
            print(f"FAILED for {doc.filename}: {e}")
            traceback.print_exc()

# Print latest statuses
docs = db.query(DocumentMetadata).all()
for d in docs[-5:]:
    print(d.filename, '|', d.risk_level, '|', d.risk_description)
