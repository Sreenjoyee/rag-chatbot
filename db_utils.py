import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import bcrypt
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# bcrypt password hashing.
def _hash_password(password: str) -> str:
    # bcrypt has a 72-byte password limit; truncate defensively.
    pw_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    pw_bytes = password.encode("utf-8")[:72]
    try:
        return bcrypt.checkpw(pw_bytes, hashed.encode("utf-8"))
    except ValueError:
        return False


# ---------- Models ----------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ApplicationLog(Base):
    __tablename__ = "application_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, index=True)
    user_query = Column(Text)
    gpt_response = Column(Text)
    model = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DocumentStore(Base):
    __tablename__ = "document_store"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String)
    upload_timestamp = Column(DateTime(timezone=True), server_default=func.now())


def create_tables():
    """Create tables if they don't exist. Called on app startup."""
    Base.metadata.create_all(bind=engine)


# ---------- User operations ----------
def create_user(username: str, password: str) -> Optional[User]:
    """Create a new user. Returns the user, or None if username is taken."""
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            return None
        user = User(username=username, password_hash=_hash_password(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Return the user if credentials are valid, else None."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            return None
        if not _verify_password(password, user.password_hash):
            return None
        return user
    finally:
        db.close()


def get_user_by_id(user_id: int) -> Optional[User]:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()


# ---------- Chat log operations (now scoped by user_id) ----------
def insert_application_logs(
    user_id: int, session_id: str, user_query: str, gpt_response: str, model: str
):
    db = SessionLocal()
    try:
        log = ApplicationLog(
            user_id=user_id,
            session_id=session_id,
            user_query=user_query,
            gpt_response=gpt_response,
            model=model,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()


def get_chat_history(user_id: int, session_id: str):
    """Return chat history for (user, session). Session IDs are only meaningful
    in the context of a user — another user's session_id won't return anything."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ApplicationLog)
            .filter(
                ApplicationLog.user_id == user_id,
                ApplicationLog.session_id == session_id,
            )
            .order_by(ApplicationLog.created_at)
            .all()
        )
        messages = []
        for row in rows:
            messages.extend(
                [
                    {"role": "human", "content": row.user_query},
                    {"role": "ai", "content": row.gpt_response},
                ]
            )
        return messages
    finally:
        db.close()


# ---------- Document record operations (now scoped by user_id) ----------
def insert_document_record(user_id: int, filename: str) -> int:
    db = SessionLocal()
    try:
        doc = DocumentStore(user_id=user_id, filename=filename)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc.id
    finally:
        db.close()


def delete_document_record(user_id: int, file_id: int) -> bool:
    """Only deletes if the document belongs to the requesting user."""
    db = SessionLocal()
    try:
        doc = (
            db.query(DocumentStore)
            .filter(DocumentStore.id == file_id, DocumentStore.user_id == user_id)
            .first()
        )
        if doc is None:
            return False
        db.delete(doc)
        db.commit()
        return True
    finally:
        db.close()


def get_user_documents(user_id: int):
    """Return only this user's documents."""
    db = SessionLocal()
    try:
        docs = (
            db.query(DocumentStore)
            .filter(DocumentStore.user_id == user_id)
            .order_by(DocumentStore.upload_timestamp.desc())
            .all()
        )
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "upload_timestamp": d.upload_timestamp,
            }
            for d in docs
        ]
    finally:
        db.close()


def document_belongs_to_user(user_id: int, file_id: int) -> bool:
    db = SessionLocal()
    try:
        doc = (
            db.query(DocumentStore)
            .filter(DocumentStore.id == file_id, DocumentStore.user_id == user_id)
            .first()
        )
        return doc is not None
    finally:
        db.close()