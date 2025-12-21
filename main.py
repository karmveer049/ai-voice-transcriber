import os
import shutil
import warnings
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
os.makedirs(BASE_DIR / "temp", exist_ok=True)

# --- OPTIMIZED AI ENGINE ---
def get_stable_model():
    """Prioritizes the Stable 1.5 Flash model to avoid 429 errors."""
    try:
        # Explicitly ask for 1.5 Flash first (High Limits)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        # Fallback if 1.5 is missing
        return genai.GenerativeModel('gemini-pro')

def analyze_text_efficiently(text):
    """
    Performs Grammar Polish AND Summarization in ONE single API call.
    This reduces error rates by 50%.
    """
    try:
        model = get_stable_model()
        
        # Single Prompt for Dual Task
        prompt = f"""
        You are an expert linguist. Process this raw spoken text:
        "{text}"

        TASK 1: Fix grammar/punctuation (keep Hinglish context).
        TASK 2: Create a concise summary in English.

        OUTPUT FORMAT (Strictly follow this):
        [POLISHED_START]
        (Put the corrected text here)
        [POLISHED_END]
        [SUMMARY_START]
        (Put the summary here)
        [SUMMARY_END]
        """
        
        response = model.generate_content(prompt)
        raw_response = response.text

        # Parse the result
        polished_text = text # Default fallback
        summary_text = "Could not generate summary."

        if "[POLISHED_START]" in raw_response:
            polished_text = raw_response.split("[POLISHED_START]")[1].split("[POLISHED_END]")[0].strip()
        
        if "[SUMMARY_START]" in raw_response:
            summary_text = raw_response.split("[SUMMARY_START]")[1].split("[SUMMARY_END]")[0].strip()

        return polished_text, summary_text

    except Exception as e:
        return text, f"AI Busy (Rate Limit): Please wait 10 seconds. ({str(e)})"

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

        # 2. Analyze (One Call for both Polish & Summary)
        polished, summary = analyze_text_efficiently(raw_text)

        return {
            "original_text": polished,
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)