# Musbi - AI Tech Guide Chatbot

Musbi is a fully functional AI chatbot designed to help students learn about Information Technology (IT) and explore tech career paths. Musbi acts as a technical professional, providing expert guidance, explaining complex concepts, and answering questions based on its internal knowledge or uploaded documents.

## 🚀 Features

- **Expert Identity**: Musbi is an IT professional who provides clear, professional, and encouraging tech advice.
- **Knowledge Base Upload**: Students and teachers can upload `.pdf` or `.txt` documents to provide Musbi with specific context or learning materials.
- **Modern UI**: A clean, responsive, and glassmorphic chat interface.
- **OpenRouter Integration**: Powered by advanced LLMs via OpenRouter (defaults to GPT-3.5-Turbo).
- **Deployment Ready**: Fully configured for local development with `uv` and deployment on **Vercel**.

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
- **AI Engine**: [OpenRouter](https://openrouter.ai/) (OpenAI SDK compatible)
- **Document Processing**: `PyPDF2`
- **Dependency Management**: `uv`

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### 2. Environment Configuration
Create a `.env` file in the root directory and add your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_api_key_here
```

### 3. Install Dependencies
Using `uv`:
```bash
uv sync
```
Or using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
uv run uvicorn main:app --reload
```
Access the chatbot at: [http://localhost:8000](http://localhost:8000)

## 🌐 Deployment (Vercel)

Musbi is pre-configured for Vercel.

1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the project root.
3. Add your `OPENROUTER_API_KEY` in the Vercel Dashboard under **Environment Variables**.

## 📖 How to Use

1. **Chat**: Type your IT questions in the input field and press Enter or click "Send".
2. **Identity**: Ask "Who are you?" or "What is your name?" to meet Musbi.
3. **Upload Knowledge**: Click the **📁 Upload Knowledge** button to add a PDF or Text file. Musbi will immediately begin using that information to answer your questions.

---
Built with ❤️ for learners and future tech professionals.
