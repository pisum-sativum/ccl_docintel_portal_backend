import os
from sqlalchemy import text
from database import engine

def migrate_rbac():
    print("Running RBAC DB migration...")
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE documents ADD COLUMN access_level VARCHAR DEFAULT 'Internal'"))
            print("Added access_level column.")
        except Exception as e:
            print(f"access_level column may already exist: {e}")

        try:
            conn.execute(text("ALTER TABLE documents ADD COLUMN department VARCHAR"))
            print("Added department column.")
        except Exception as e:
            print(f"department column may already exist: {e}")

        try:
            conn.execute(text("ALTER TABLE documents ADD COLUMN doc_type VARCHAR"))
            print("Added doc_type column.")
        except Exception as e:
            print(f"doc_type column may already exist: {e}")

        try:
            conn.execute(text("ALTER TABLE documents ADD COLUMN summary TEXT"))
            print("Added summary column.")
        except Exception as e:
            print(f"summary column may already exist: {e}")
            
        conn.commit()
    print("Migration finished.")

if __name__ == "__main__":
    migrate_rbac()
