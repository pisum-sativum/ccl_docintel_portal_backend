import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load key-value properties out of our secured .env file 
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("WARNING: DATABASE_URL is not set. Using temporary sqlite database.")
    DATABASE_URL = "sqlite:///./sql_app.db"
# Connect to cloud PostgreSQL server (Neon)
connect_args = {"sslmode": "require"} if DATABASE_URL.startswith("postgres") else {"check_same_thread": False}
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,   # detect stale connections automatically
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DocumentMetadata(Base):
    __tablename__ = "documents"

    id           = Column(Integer, primary_key=True, index=True)
    filename     = Column(String, unique=True, index=True)
    content_type = Column(String)
    char_count   = Column(Integer)
    extracted_text = Column(Text, nullable=True)   # full extracted content
    upload_date  = Column(DateTime, default=datetime.datetime.utcnow)
    risk_level = Column(String, default="None")
    risk_description = Column(String, default="Pending scan.")
    content_hash = Column(String, nullable=True, index=True)
    access_level = Column(String, default="Internal")
    department = Column(String, nullable=True)
    doc_type = Column(String, nullable=True)
    summary = Column(Text, nullable=True)

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role            = Column(String, default="viewer")  # 'admin' | 'viewer'
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()