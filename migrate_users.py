"""
Migration: creates the 'users' table in Neon PostgreSQL.
Run once: python migrate_users.py
"""
from database import engine, Base, User

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Migration complete: 'users' table created (if it didn't already exist).")
