from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import os
import shutil
from contextlib import asynccontextmanager

model_path = r"D:\D\Dysarthria\ASR\models\base.pt"
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    print("Loading Whisper model...")
    models["base"] = whisper.load_model(model_path)
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 1. Validation: Ensure it's an audio file
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File provided is not audio.")

    # 2. Save the uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 3. Pass the file path to the Whisper model
        # Whisper handles the heavy lifting of resampling and decoding
        result = models["base"].transcribe(temp_file)
        
        return {
            "filename": file.filename,
            "text": result["text"],
            "language": result.get("language")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 4. Cleanup: Remove the temp file after processing
        if os.path.exists(temp_file):
            os.remove(temp_file)