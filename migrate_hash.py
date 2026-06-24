"""
Migration to add content_hash column to the documents table.
"""
from database import engine

MIGRATION_SQL = """
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS content_hash VARCHAR;
CREATE INDEX IF NOT EXISTS ix_documents_content_hash ON documents(content_hash);
"""

if __name__ == "__main__":
    with engine.connect() as conn:
        conn.execute(__import__("sqlalchemy").text(MIGRATION_SQL))
        conn.commit()
    print("Migration complete: content_hash column added.")
