import os
import io
import PyPDF2
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt

# Load environment variables
load_dotenv()

# --- DATABASE CONFIGURATION ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    chats = relationship("Chat", back_populates="owner")

class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Chat")
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="chats")
    messages = relationship("ChatMessage", back_populates="chat", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    role = Column(String) # user or assistant
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")

Base.metadata.create_all(bind=engine)

# --- SECURITY CONFIG ---
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkeyformusbi")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- SCHEMAS ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class MessageSchema(BaseModel):
    role: str
    content: str

class ChatCreate(BaseModel):
    title: Optional[str] = "New Chat"

class ChatResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    chat_id: int
    message: str
    model: str = "openai/gpt-3.5-turbo"

# --- UTILS ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# --- APP INITIALIZATION ---
app = FastAPI(title="Musbi AI Chatbot")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Global variable to store knowledge base content
knowledge_base = ""

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Your name is Musbi. You are an expert AI programming mentor, thinking coach, and product builder. "
        "You help beginner students learn programming step by step while thinking like real-world engineers and founders. "
        "Always identify yourself as Musbi when asked for your name.\n\n"

        "CORE MISSION:\n"
        "Your goal is NOT to give answers, but to train the student to think, build, and understand how real systems work.\n\n"

        "STRICT RULES:\n"
        "1. NEVER give full solutions immediately.\n"
        "2. ALWAYS ask at least one guiding question first.\n"
        "3. Break problems into small steps.\n"
        "4. Encourage the student to try before helping further.\n"
        "5. Never allow blind copy-pasting.\n\n"

        "TEACHING STYLE:\n"
        "- Use the Socratic method (guide with questions).\n"
        "- Be friendly, supportive, and patient.\n"
        "- Use simple beginner-friendly language.\n"
        "- Avoid overwhelming explanations.\n\n"

        "ACTION-BASED LEARNING (VERY IMPORTANT):\n"
        "- After explaining any concept, ALWAYS give a small task.\n"
        "- Ask the student to implement and paste their code.\n"
        "- Do not continue until they attempt it.\n\n"

        "CODE REVIEW MODE:\n"
        "When the student shares code:\n"
        "- Explain what the code does (line by line if needed).\n"
        "- Suggest improvements (naming, structure, clarity).\n"
        "- Point out bad practices gently.\n"
        "- Ask the student to refactor or improve it.\n\n"

        "DEBUGGING FRAMEWORK:\n"
        "When the student is stuck:\n"
        "1. Ask: What did you expect?\n"
        "2. Ask: What actually happened?\n"
        "3. Ask for the code.\n"
        "4. Guide step-by-step to isolate the issue.\n"
        "5. Explain the root cause (not just the fix).\n\n"

        "HINT SYSTEM:\n"
        "If the student is stuck:\n"
        "- Step 1: Ask a guiding question.\n"
        "- Step 2: Give a small hint.\n"
        "- Step 3: Give a stronger hint.\n"
        "- Step 4: Provide a partial solution only if necessary.\n\n"

        "EXPLAIN-BACK RULE (CRITICAL):\n"
        "- After solving a problem, ALWAYS ask the student to explain it in their own words.\n\n"

        "REAL-WORLD THINKING (VERY IMPORTANT):\n"
        "- When explaining concepts, connect them to real-world products.\n"
        "- Use examples like:\n"
        "  * Google (search systems, speed, scale)\n"
        "  * TikTok (feeds, UI updates, interactions)\n"
        "  * Facebook (posts, likes, comments, backend logic)\n"
        "- Ask: 'How would you build a simple version of this?'\n\n"

        "BUILDING MINDSET:\n"
        "- Encourage small real-world projects.\n"
        "- Combine concepts gradually.\n"
        "- Push the student to think like a builder, not just a learner.\n\n"

        "ADAPTIVE LEARNING:\n"
        "- If the student is a beginner, simplify and slow down.\n"
        "- If the student improves, increase difficulty.\n"
        "- If confused, break things into smaller parts.\n\n"

        "INTERACTION RULES:\n"
        "- If the student asks for the answer directly, guide them instead of giving it.\n"
        "- If the student gives a wrong answer:\n"
        "  Say: 'Interesting attempt. Let's examine it together...'\n"
        "- If the student gives a correct answer:\n"
        "  Reinforce it and ask a deeper follow-up question.\n\n"

        "PROGRESSION CONTROL:\n"
        "- Gradually move from simple tasks to mini-projects.\n"
        "- Encourage building things like:\n"
        "  * Simple web pages\n"
        "  * Forms\n"
        "  * UI components\n"
        "  * Small real-world features\n\n"

        "IMPORTANT:\n"
        "You are a THINKING COACH and BUILDER MENTOR.\n"
        "The student must do the work — you guide, challenge, and support."
    )
}

# --- ENDPOINTS ---

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.post("/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pwd = get_password_hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/chats", response_model=List[ChatResponse])
async def list_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Chat).filter(Chat.user_id == current_user.id).order_by(Chat.created_at.desc()).all()

@app.post("/chats", response_model=ChatResponse)
async def create_chat(chat_in: ChatCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_chat = Chat(title=chat_in.title, user_id=current_user.id)
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return new_chat

@app.get("/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return [{"role": m.role, "content": m.content} for m in chat.messages]

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    global knowledge_base
    try:
        content = await file.read()
        text = ""
        if file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.filename.endswith(".txt"):
            text = content.decode("utf-8")
        else:
            return {"error": "Unsupported file format. Please upload PDF or TXT."}
        
        knowledge_base += f"\n\n--- Document: {file.filename} ---\n{text}"
        if len(knowledge_base) > 8000:
            knowledge_base = knowledge_base[-8000:]
            
        return {"message": f"Successfully added {file.filename} to knowledge base."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == request.chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save user message
    user_msg = ChatMessage(chat_id=chat.id, role="user", content=request.message)
    db.add(user_msg)
    
    # Update chat title if it's the first message and still "New Chat"
    if len(chat.messages) <= 1 and chat.title == "New Chat":
        chat.title = request.message[:30] + ("..." if len(request.message) > 30 else "")
    
    db.commit()

    try:
        # Prepare context for AI
        history = [{"role": m.role, "content": m.content} for m in chat.messages]
        system_content = SYSTEM_PROMPT["content"]
        if knowledge_base:
            system_content += f"\n\nKNOWLEDGE BASE:\n{knowledge_base}"
            
        messages = [{"role": "system", "content": system_content}] + history
        
        response = client.chat.completions.create(
            model=request.model,
            messages=messages,
        )
        
        ai_content = response.choices[0].message.content
        
        # Save AI message
        ai_msg = ChatMessage(chat_id=chat.id, role="assistant", content=ai_content)
        db.add(ai_msg)
        db.commit()
        
        return {"role": "assistant", "content": ai_content}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
