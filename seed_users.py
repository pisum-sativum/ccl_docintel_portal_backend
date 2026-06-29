"""
Seeds the 'users' table with default admin, operator, and viewer accounts.
Run once after migrate_users.py: python seed_users.py
"""

from auth import hash_password
from database import SessionLocal, User

SEED_USERS = [
    {"username": "admin", "password": "admin123", "role": "admin"},
    {"username": "operator", "password": "operator123", "role": "operator"},
    {"username": "viewer", "password": "viewer123", "role": "viewer"},
]


def seed():
    db = SessionLocal()
    created = 0
    for u in SEED_USERS:
        exists = db.query(User).filter(User.username == u["username"]).first()
        if not exists:
            new_user = User(
                username=u["username"],
                hashed_password=hash_password(u["password"]),
                role=u["role"],
            )
            db.add(new_user)
            created += 1
            print(f"Created user: {u['username']} (role={u['role']})")
        else:
            print(f"User '{u['username']}' already exists — skipping.")
    db.commit()
    db.close()
    print(f"\nSeeding complete. {created} new user(s) created.")


if __name__ == "__main__":
    seed()
