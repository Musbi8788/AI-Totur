import os
import io
import PyPDF2
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Teacher Chatbot")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Global variable to store knowledge base content
knowledge_base = ""

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "openai/gpt-3.5-turbo" # Default model for OpenRouter

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Your name is Musbi. You are a highly skilled technical professional and IT expert. "
        "You are interacting with students to help them learn about IT and related career paths. "
        "Always identify yourself as Musbi when asked for your name. "
        "Your goal is to provide expert technical guidance, explain IT concepts clearly, and offer career advice in the tech industry. "
        "Be encouraging, professional, and knowledgeable. "
        "If context is provided under 'KNOWLEDGE BASE', use it to answer questions accurately. "
        "If you don't know the answer, say so, but try to guide the student based on your technical expertise."
    )
}

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {"status": "online", "message": "Musbi is ready to help!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
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
        
        # Append to knowledge base (simple version)
        knowledge_base += f"\n\n--- Document: {file.filename} ---\n{text}"
        # Keep it within reasonable limits for simple prompt injection (approx 4000 chars)
        if len(knowledge_base) > 8000:
            knowledge_base = knowledge_base[-8000:]
            
        return {"message": f"Successfully added {file.filename} to knowledge base."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Prepare messages with system prompt and knowledge base
        system_content = SYSTEM_PROMPT["content"]
        if knowledge_base:
            system_content += f"\n\nKNOWLEDGE BASE:\n{knowledge_base}"
            
        messages = [{"role": "system", "content": system_content}] + [m.model_dump() for m in request.messages]
        
        response = client.chat.completions.create(
            model=request.model,
            messages=messages,
        )
        
        return response.choices[0].message
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
