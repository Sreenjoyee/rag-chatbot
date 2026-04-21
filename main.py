import os
import uuid
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Request,
    Depends,
    Form,
    status,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import (
    create_tables,
    create_user,
    authenticate_user,
    get_user_by_id,
    insert_application_logs,
    get_chat_history,
    get_user_documents,
    insert_document_record,
    delete_document_record,
    document_belongs_to_user,
    User,
)
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma

load_dotenv()

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError(
        "SESSION_SECRET environment variable is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(lifespan=lifespan)

# Signed session cookies (backed by itsdangerous).
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- Auth helpers ----------
def get_current_user(request: Request) -> User:
    """Dependency: require a logged-in user. Raises 401 if not logged in."""
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = get_user_by_id(user_id)
    if user is None:
        # Stale cookie pointing at a deleted user
        request.session.clear()
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def get_optional_user(request: Request) -> Optional[User]:
    """Like get_current_user but returns None instead of raising.
    Used on pages that render differently when not logged in."""
    user_id = request.session.get("user_id")
    if user_id is None:
        return None
    return get_user_by_id(user_id)


# ---------- Auth pages ----------
@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse(
        request, "signup.html", {"error": error, "current_user": None}
    )


@app.post("/signup")
def signup_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    username = username.strip()
    if len(username) < 3:
        return RedirectResponse(
            "/signup?error=Username+must+be+at+least+3+characters",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    if len(password) < 6:
        return RedirectResponse(
            "/signup?error=Password+must+be+at+least+6+characters",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    if password != confirm_password:
        return RedirectResponse(
            "/signup?error=Passwords+do+not+match",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    user = create_user(username, password)
    if user is None:
        return RedirectResponse(
            "/signup?error=Username+already+taken",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # Auto-login after signup
    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse(
        request, "login.html", {"error": error, "current_user": None}
    )


@app.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = authenticate_user(username.strip(), password)
    if user is None:
        return RedirectResponse(
            "/login?error=Invalid+username+or+password",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)


# ---------- HTML pages (auth-required) ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = get_optional_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request, "index.html", {"current_user": user}
    )


@app.get("/docs-ui", response_class=HTMLResponse)
def docs_ui(request: Request):
    user = get_optional_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    docs = get_user_documents(user.id)
    return templates.TemplateResponse(
        request, "documents.html", {"documents": docs, "current_user": user}
    )


# ---------- JSON API: chat ----------
@app.post("/chat", response_model=QueryResponse)
def chat(
    query_input: QueryInput,
    current_user: User = Depends(get_current_user),
):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(
        f"User: {current_user.username}, Session: {session_id}, "
        f"Query: {query_input.question}, Model: {query_input.model.value}"
    )

    chat_history = get_chat_history(current_user.id, session_id)
    rag_chain = get_rag_chain(query_input.model.value, current_user.id)

    try:
        answer = rag_chain.invoke(
            {"input": query_input.question, "chat_history": chat_history}
        )["answer"]
    except Exception as e:
        logging.exception(f"RAG pipeline failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "The model or embedding service is temporarily unavailable. "
                "Please try again in a moment."
            ),
        )

    insert_application_logs(
        current_user.id, session_id, query_input.question, answer,
        query_input.model.value,
    )
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


# ---------- JSON API: document upload ----------
@app.post("/upload-doc")
def upload_and_index_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    allowed_extensions = [".pdf", ".docx", ".html"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    fd, temp_file_path = tempfile.mkstemp(suffix=file_extension)
    try:
        with os.fdopen(fd, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(current_user.id, file.filename)
        success = index_document_to_chroma(temp_file_path, file_id, current_user.id)

        if success:
            return {
                "message": f"File {file.filename} uploaded and indexed.",
                "file_id": file_id,
            }
        else:
            delete_document_record(current_user.id, file_id)
            raise HTTPException(
                status_code=500, detail=f"Failed to index {file.filename}."
            )
    finally:
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except OSError as e:
            logging.warning(f"Could not remove temp file {temp_file_path}: {e}")


# ---------- JSON API: list documents ----------
@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents(current_user: User = Depends(get_current_user)):
    return get_user_documents(current_user.id)


# ---------- JSON API: delete document ----------
@app.post("/delete-doc")
def delete_document(
    request: DeleteFileRequest,
    current_user: User = Depends(get_current_user),
):
    # Ownership check first — don't even touch Chroma if the user doesn't own this.
    if not document_belongs_to_user(current_user.id, request.file_id):
        raise HTTPException(status_code=404, detail="Document not found.")

    chroma_ok = delete_doc_from_chroma(request.file_id, current_user.id)
    if not chroma_ok:
        return {"error": f"Failed to delete document {request.file_id} from Chroma."}

    db_ok = delete_document_record(current_user.id, request.file_id)
    if not db_ok:
        return {
            "error": f"Deleted from Chroma but failed to delete DB record for {request.file_id}."
        }
    return {"message": f"Successfully deleted document {request.file_id}."}