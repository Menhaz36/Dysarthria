
import os
import platform

# ── MUST be set before any huggingface/transformers imports ──────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_XET"] = "1"

if platform.system() == "Windows":
    base = r"D:\D\IMPORTANT\Dysarthria\ASR\huggingface_cache"
else:
    base = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    )

os.environ["HF_HOME"] = base
os.environ["TRANSFORMERS_CACHE"] = f"{base}/hub"
os.environ["HF_DATASETS_CACHE"] = f"{base}/datasets"
os.environ["TORCH_HOME"] = f"{base}/torch"

import gc
import asyncio

import sys
import torch # type: ignore
import numpy as np
from transformers import pipeline, WhisperFeatureExtractor # type: ignore
from langchain_core.runnables import RunnableLambda # type: ignore
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_groq import ChatGroq # type: ignore
from huggingface_hub import login # type: ignore
from pydantic import SecretStr
from dotenv import load_dotenv

print("Starting ASR pipeline setup (Fine-tuned whisper-tiny on TORGO)...")
load_dotenv()

# ── Auth ──────────────────────────────────────────────────────────────────────
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_HUB_DISABLE_XET"] = "1"
if hf_token:
    print("Logging into Hugging Face Hub...")
    login(token=hf_token)

# ── Device selection ──────────────────────────────────────────────────────────
gc.collect()
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32          # MPS is unstable with fp16 on tiny models
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = 0
    torch_dtype = torch.float32          # whisper-tiny is small; fp32 is safe & avoids NaN
    print("Using CUDA GPU")
else:
    device = -1
    torch_dtype = torch.float32
    print("Using CPU")

# ── Model ─────────────────────────────────────────────────────────────────────
# whisper-tiny based fine-tune: word-level timestamps are NOT supported.
# We use return_timestamps=True (segment-level) to still get pause info where available,
# and fall back gracefully when timestamps are absent.

MODEL_ID = "ruch9265/distil-whisper-torgo"   # ← your fine-tuned repo

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    MODEL_ID,
    feature_size=80,       # force correct value — model weights expect 80
    sampling_rate=16000,
)

print(f"Loading ASR model: {MODEL_ID} ...")
asr_model = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    feature_extractor=feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
    return_timestamps=True,              # segment-level; word-level not available on tiny
    generate_kwargs={"language": "english", "task": "transcribe"},
)
print("Fine-tuned model loaded successfully.")


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe(audio_file: str) -> dict:
    """Run ASR on a file path and return the raw Whisper output dict."""
    try:
        result = asr_model(audio_file)
        return result
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise


async def async_transcribe(audio_file: str) -> dict:
    """Non-blocking wrapper so the FastAPI event loop stays free."""
    return await asyncio.to_thread(transcribe, audio_file)


# ── Pause-aware formatter ─────────────────────────────────────────────────────
def format_pause_aware_transcript(
    whisper_result: dict,
    pause_threshold: float = 0.8,
) -> dict:
    """
    Build a transcript string that annotates silences ≥ pause_threshold seconds.

    Works with BOTH segment-level chunks (whisper-tiny) and word-level chunks
    (distil-large-v3).  Falls back to plain text when no timestamp data exists.
    """
    chunks = whisper_result.get("chunks", [])

    # ── Fallback: no chunk data → return raw text ──────────────────────────
    if not chunks:
        plain = whisper_result.get("text", "").strip()
        print("[pause_formatter] No chunk data — returning plain transcript.")
        return {"input": plain}

    # ── Check whether timestamps are present and numeric ──────────────────
    def _valid_ts(chunk):
        ts = chunk.get("timestamp")
        return (
            ts is not None
            and len(ts) == 2
            and ts[0] is not None
            and ts[1] is not None
        )

    has_timestamps = any(_valid_ts(c) for c in chunks)

    if not has_timestamps:
        # Stitch text without pause markers
        plain = " ".join(c.get("text", "").strip() for c in chunks).strip()
        print("[pause_formatter] Timestamps absent — returning plain transcript.")
        return {"input": plain}

    # ── Build pause-annotated string ───────────────────────────────────────
    formatted_parts: list[str] = []

    for i, chunk in enumerate(chunks):
        word = chunk.get("text", "").strip()
        if not word:
            continue

        if i > 0 and _valid_ts(chunks[i - 1]) and _valid_ts(chunk):
            gap = chunk["timestamp"][0] - chunks[i - 1]["timestamp"][1]
            if gap >= pause_threshold:
                formatted_parts.append(f"[{gap:.1f}s pause]")

        formatted_parts.append(word)

    final_string = " ".join(formatted_parts)
    return {"input": final_string}


# ── LangChain runnables ───────────────────────────────────────────────────────
transcribe_runnable = RunnableLambda(async_transcribe)
pause_formatter_runnable = RunnableLambda(format_pause_aware_transcript)

# ── Few-shot examples ─────────────────────────────────────────────────────────
examples = [
    {
        "input": "I [2.5s pause] go [1.2s pause] store [2.0s pause] milk",
        "output": "I am going to the store to get some milk.",
    },
    {
        "input": "Want [3.0s pause] water [1.0s pause] cold",
        "output": "I want a glass of cold water.",
    },
    {
        "input": "Turn [1.5s pause] light off [2.5s pause] room",
        "output": "Please turn off the lights in the room.",
    },
    {
        "input": "My name [0.5s pause] is [4.0s pause] John",
        "output": "My name is John.",
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Transcript: {input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

SYSTEM_PROMPT = """
You are an AI Speech Transcript Repair Assistant.

You receive automatic speech recognition transcripts from a speaker with a speech impairment.

Pauses marked like [2.0s pause] may indicate where short functional words
(e.g., articles, auxiliary verbs, prepositions) were unintentionally omitted.

Your task is to minimally repair the sentence so it becomes grammatically correct
while preserving the speaker's original wording, meaning, and intent.

Repair Rules:
1. Only insert or adjust small functional words when clearly necessary
2. Do NOT paraphrase, summarize, or restructure the sentence
3. Do NOT add new information or interpretations
4. If the sentence is already correct, return it unchanged
5. Ignore pause markers in the final output
6. Remove simple speech disfluencies:
   6.1 repeated whole words (e.g., "I I want" → "I want")
   6.2 filler sounds (e.g., "um", "uh")
7. Do NOT alter emphasis or stylistic repetition used intentionally
8. Preserve original sentence boundaries; do not merge or split sentences

Output Rules:
• Output exactly one corrected sentence
• No explanations
• No commentary
• No quotation marks
"""

final_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    few_shot_prompt,
    ("human", "Transcript: {input}"),
])

# ── LLM ───────────────────────────────────────────────────────────────────────
groq_api_key = SecretStr(os.getenv("groq_api_key") or "")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=groq_api_key,
)

# ── Full pipeline ─────────────────────────────────────────────────────────────
speech_repair_chain = (
    transcribe_runnable
    | pause_formatter_runnable
    | final_prompt
    | llm
    | StrOutputParser()
)


async def process_audio(file_path: str) -> str:
    """Entry point called by main1.py / FastAPI."""
    return await speech_repair_chain.ainvoke(file_path)


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"D:\D\IMPORTANT\Dysarthria\ASR\Sample_dysarthria_audio.wav"
    )
    result = asyncio.run(process_audio(path))
    print("Final Repaired Transcript:", result)