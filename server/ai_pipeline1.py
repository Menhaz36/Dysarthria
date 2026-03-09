from tqdm import asyncio
from transformers import pipeline
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os, tempfile
import gc
from datasets import load_dataset, Audio
import soundfile as sf
import numpy as np
import io
import torch
import asyncio



print("Starting ASR pipeline setup...")
load_dotenv() 

# Clear caches before loading
print("Clearing caches...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


device =0 if torch.cuda.is_available() else -1
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Using GPU: {torch.cuda.is_available()} using {torch_dtype}" if device == 0 else "Using CPU")




os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

base = "D:/D/Dysarthria/ASR/huggingface_cache"

os.environ["HF_HOME"] = base
os.environ["TRANSFORMERS_CACHE"] = f"{base}/hub"
os.environ["HF_DATASETS_CACHE"] = f"{base}/datasets"
os.environ["TORCH_HOME"] = f"{base}/torch"

# --- Model Initialization ---
print("Loading ASR model...")
asr_model=pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/distil-large-v3",
        device=device,
        torch_dtype=torch_dtype,
        # return_timestamps="word" is critical to get the start/end of every single word
        return_timestamps="word",
)
print("Model loaded successfully.")


def transcribe(audio_file):
    try:
        result=asr_model(audio_file)
        return result #not necessary result will be a dict with 'text' key but it is the case for whisper-base
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise e

async def async_transcribe(audio_file):
    """Asynchronous wrapper for LangChain"""
    # Offload the blocking PyTorch computation to a separate thread
    return await asyncio.to_thread(transcribe, audio_file)

def format_pause_aware_transcript(whisper_result: dict, pause_threshold: float = 0.8) -> dict:
    """
    Runnable node: Takes raw Whisper output, calculates gaps between words,
    and returns a dictionary ready for the LangChain Prompt.
    """
    chunks = whisper_result.get("chunks", [])
    formatted_text = []
    
    for i in range(len(chunks)):
        current_chunk = chunks[i]
        word = current_chunk["text"].strip()
        
        if i > 0:
            prev_chunk = chunks[i-1]
            # Gap = Current word start - Previous word end
            current_start = current_chunk["timestamp"][0]
            prev_end = prev_chunk["timestamp"][1]
            
            # Handle edge cases where timestamps might be None
            if current_start is not None and prev_end is not None:
                gap = current_start - prev_end
                if gap >= pause_threshold:
                    formatted_text.append(f"[{gap:.1f}s pause]")
                
        formatted_text.append(word)
        
    final_string = " ".join(formatted_text)
    
    # Runnables must return dictionaries that match the expected variables in the prompt
    return {"input": final_string}

# --- LangChain Setup ---
'''If you want to keep async_transcribe, RunnableLambda supports async functions when passed directly (not wrapped in a lambda):
python# ✅ Correct — pass the async function directly, not via lambda
transcribe_runnable = RunnableLambda(async_transcribe)
The lambda wrapper lambda audio_file: async_transcribe(audio_file) is itself sync — it returns a coroutine object without awaiting it. Passing the async function directly lets LangChain detect it as a coroutine and await it properly.
'''
# transcribe_runnable=RunnableLambda(lambda audio_file: async_transcribe(audio_file))
# or
transcribe_runnable = RunnableLambda(async_transcribe)    # async version, no lambda wrapper
pause_formatter_runnable = RunnableLambda(format_pause_aware_transcript)

examples = [
    {
        "input": "I [2.5s pause] go [1.2s pause] store [2.0s pause] milk",
        "output": "I am going to the store to get some milk."
    },
    {
        "input": "Want [3.0s pause] water [1.0s pause] cold",
        "output": "I want a glass of cold water."
    },
    {
        "input": "Turn [1.5s pause] light off [2.5s pause] room",
        "output": "Please turn off the lights in the room."
    },
    {
        "input": "My name [0.5s pause] is [4.0s pause] John",
        "output": "My name is John." 
    }
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Transcript: {input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


SYSTEM_PROMPT =""" 
You are an AI Speech Transcript Repair Assistant.

You receive automatic speech recognition transcripts from a speaker with a speech impairment.

Pauses marked like [2.0s pause] may indicate where short functional words 
(e.g., articles, auxiliary verbs, prepositions) were unintentionally omitted.

Your task is to minimally repair the sentence so it becomes grammatically correct 
while preserving the speaker’s original wording, meaning, and intent.

Repair Rules:
1. Only insert or adjust small functional words when clearly necessary
2. Do NOT paraphrase, summarize, or restructure the sentence
3. Do NOT add new information or interpretations
4. If the sentence is already correct, return it unchanged
5. Ignore pause markers in the final output
6. Remove simple speech disfluencies:
6.1 repeated whole words (e.g., “I I want” → “I want”)
6.2 filler sounds (e.g., “um”, “uh”)
7. Do NOT alter emphasis or stylistic repetition used intentionally
8. If the transcript is already correct, return it unchanged
9. Ignore pause markers like [2.0s pause] in the final output
10. Preserve original sentence boundaries; do not merge or split sentences

Output Rules:
• Output exactly one corrected sentence
• No explanations
• No commentary
• No quotation marks
"""


final_prompt=ChatPromptTemplate.from_messages([
      ("system",SYSTEM_PROMPT),
      few_shot_prompt,
      ("human", "Transcript: {input}")
])

# # --- Groq Setup ---
# groq_api_key = os.getenv("groq_api_key")
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.0, 
)

# --- Final Pipeline ---

speech_repair_chain = (
    transcribe_runnable
    |pause_formatter_runnable 
    | final_prompt 
    | llm 
    | StrOutputParser()
)

async def process_audio(file_path: str) -> str:
    """Entry point for the FastAPI server to trigger the async pipeline."""
    # Use .ainvoke() instead of .invoke() for asynchronous execution
    return await speech_repair_chain.ainvoke(file_path)


if __name__ == "__main__":
    file_path=r"D:\D\Dysarthria\ASR\audio.mp3"
    # Use asyncio.run() to execute the top-level async function
    result = asyncio.run(process_audio(file_path))
    print("Final Repaired Transcript:", result)
