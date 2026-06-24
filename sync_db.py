import os
from database import SessionLocal, DocumentMetadata
from rag_engine import vector_db

def sync_database():
    db = SessionLocal()
    docs = db.query(DocumentMetadata).all()
    deleted_count = 0
    
    for doc in docs:
        file_path = os.path.join("documents", doc.filename)
        # If the file is missing from the local disk folder
        if not os.path.exists(file_path):
            print(f"File '{doc.filename}' missing from disk. Deleting from DB and Vector Store...")
            
            # 1. Delete from ChromaDB Vector Store
            try:
                existing = vector_db._collection.get(where={"source": doc.filename})
                if existing and existing.get("ids"):
                    vector_db._collection.delete(ids=existing["ids"])
            except Exception as e:
                print(f"  -> Error removing from vector db: {e}")
            
            # 2. Delete from Neon PostgreSQL Database
            db.delete(doc)
            deleted_count += 1
    
    db.commit()
    db.close()
    print(f"Sync complete. Removed {deleted_count} orphaned records from the database.")

if __name__ == "__main__":
    sync_database()
