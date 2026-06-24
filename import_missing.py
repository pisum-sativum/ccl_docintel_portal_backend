import os
import hashlib
import re
from database import SessionLocal, DocumentMetadata
from extractor import extract_text_from_file
from rag_engine import inject_text_into_vector_store, scan_text_for_compliance_risks

UPLOAD_DIR = "documents"

def import_missing():
    db = SessionLocal()
    existing_docs = db.query(DocumentMetadata).all()
    existing_filenames = {d.filename for d in existing_docs}
    
    files_on_disk = os.listdir(UPLOAD_DIR)
    imported = 0
    
    for filename in files_on_disk:
        if filename not in existing_filenames:
            file_path = os.path.join(UPLOAD_DIR, filename)
            print(f"Importing missing file: {filename}")
            
            # Extract text
            parsed_text = extract_text_from_file(file_path)
            char_count = len(parsed_text)
            
            # Hash
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
            
            # Create DB record (initially pending)
            db_record = DocumentMetadata(
                filename=filename,
                content_type="application/octet-stream", # Generic
                char_count=char_count,
                extracted_text=parsed_text[:50000],
                risk_level="Scanning...",
                risk_description="AI compliance scan in progress.",
                content_hash=file_hash
            )
            db.add(db_record)
            db.commit()
            db.refresh(db_record)
            
            # Run AI synchronously for the script
            try:
                audit_result = scan_text_for_compliance_risks(parsed_text, filename)
                inject_text_into_vector_store(parsed_text, filename)
                
                db_record.risk_level = audit_result["risk_level"]
                db_record.risk_description = audit_result["description"]
                db.commit()
                print(f"  -> Imported successfully with risk: {audit_result['risk_level']}")
            except Exception as e:
                print(f"  -> AI scan failed: {e}")
            imported += 1

    db.close()
    print(f"Imported {imported} missing files.")

if __name__ == "__main__":
    import_missing()
