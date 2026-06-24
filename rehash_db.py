import os
import hashlib
import re
from database import SessionLocal, DocumentMetadata
from extractor import extract_text_from_file

UPLOAD_DIR = "documents"

def recompute_hashes():
    db = SessionLocal()
    docs = db.query(DocumentMetadata).all()
    updated_count = 0
    
    for doc in docs:
        file_path = os.path.join(UPLOAD_DIR, doc.filename)
        if not os.path.exists(file_path):
            continue
            
        # 1. Extract text using the latest extractor logic
        parsed_text = extract_text_from_file(file_path)
        
        # 2. Sanitize and clean the text exactly like main.py
        clean_text = re.sub(r'\[Image Metadata\].*?(?=\n\[|$)', '', parsed_text, flags=re.DOTALL)
        clean_text = re.sub(r'\[EXIF Data\].*?(?=\n\[|$)', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'\[OCR Error\].*?(?=\n\[|$)', '', clean_text, flags=re.DOTALL)
        clean_text = clean_text.strip().lower()
        
        # 3. Apply the fallback hashing logic
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
                    
        # 4. Generate the new fingerprint
        new_hash = hashlib.sha256(hash_material).hexdigest()
        
        # 5. Update if it has changed
        if doc.content_hash != new_hash:
            doc.content_hash = new_hash
            updated_count += 1
            print(f"Updated legacy hash for: {doc.filename}")
            
    db.commit()
    db.close()
    print(f"Re-hashing complete. Updated {updated_count} files to the new V2 logic.")

if __name__ == "__main__":
    recompute_hashes()
