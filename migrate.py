"""
One-time migration: adds risk_level and risk_description columns to the
documents table on the live Neon PostgreSQL database.

Run once with:
    python migrate.py
"""
from database import engine

MIGRATION_SQL = """
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS risk_level        VARCHAR DEFAULT 'None',
    ADD COLUMN IF NOT EXISTS risk_description  VARCHAR DEFAULT 'Pending scan.';
"""

if __name__ == "__main__":
    with engine.connect() as conn:
        conn.execute(__import__("sqlalchemy").text(MIGRATION_SQL))
        conn.commit()
    print("Migration complete: risk_level and risk_description columns added.")
