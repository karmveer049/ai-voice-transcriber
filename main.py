import os
import shutil
import warnings
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles # <--- 1. NEW IMPORT
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

# 0. Suppress Warnings
warnings.filterwarnings("ignore")

# 1. Load Environment Variables
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# 2. Setup API Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Initialize App
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

# --- ESSENTIAL SETUP FOR STATIC FILES (IMAGES/CSS) ---
# Ensure the static directory exists (optional safety step)
os.makedirs(BASE_DIR / "static", exist_ok=True)

# 2. MOUNT THE STATIC DIRECTORY
# This tells FastAPI: "If a request starts with /static, look in the 'static' folder"
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
os.makedirs(BASE_DIR / "temp", exist_ok=True)

# --- SMART MODEL FINDER ---
def get_working_model():
    """Finds the best available model for your specific API key."""
    try:
        # Prefer flash models for speed
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
                return genai.GenerativeModel(m.name)
        # Fallback
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return genai.GenerativeModel('gemini-pro')

def polish_text(text):
    """Uses AI to fix grammar and punctuation (Hinglish friendly)."""
    try:
        model = get_working_model()
        prompt = f"""
        Fix the punctuation and grammar of this text (keep Hinglish as is, just fix signs like '?' or '.'):
        "{text}"
        Return ONLY the corrected text.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return text # Return original if AI fails

def generate_summary(text):
    try:
        model = get_working_model()
        prompt = f"""
        Analyze this text: "{text}"
        1. Concise English Summary.
        2. Key Points.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    temp_filename = BASE_DIR / "temp" / file.filename
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Transcribe (Whisper)
        with open(temp_filename, "rb") as file_stream:
            transcription = groq_client.audio.transcriptions.create(
                file=(str(temp_filename), file_stream.read()),
                model="whisper-large-v3",
                response_format="json",
                language="hi", 
                temperature=0.0
            )
        raw_text = transcription.text

        # 2. Polish & Summarize (Gemini)
        polished_text = polish_text(raw_text)
        summary_text = generate_summary(polished_text)

        return {
            "original_text": polished_text,
            "summary": summary_text
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)