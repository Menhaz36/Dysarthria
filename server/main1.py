from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os

from ai_pipeline1 import process_audio

app = FastAPI(title="Dysarthria ASR Speech Repair API")

@app.post("/api/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    temp_file_path = None
    try:
        # 1. Create a temporary file to hold the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # 2. Run your LangChain pipeline
        repaired_text = await process_audio(temp_file_path)

        # 3. Clean up the temp file
        os.remove(temp_file_path)

        # 4. Return the result
        return {
            "filename": audio_file.filename,
            "text": repaired_text,
            "language": "en"  # Placeholder, you can implement language detection if needed
        }
    
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the ASR API. Use /transcribe endpoint to transcribe audio files."}

