import streamlit as st
import requests

st.set_page_config(page_title="Dysarthria ASR Interface", page_icon="🎙️")
st.title("🎙️ Whisper ASR for Dysarthria Research")
st.markdown("Upload an audio file or record directly from your microphone.")

BACKEND_URL = "http://127.0.0.1:8000/api/transcribe"

# --- Input Method Tabs ---
tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Record Audio"])

audio_to_process = None
audio_filename = None
audio_mimetype = None

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "m4a"]
    )
    if uploaded_file is not None:
        st.audio(uploaded_file)
        audio_to_process = uploaded_file.getvalue()
        audio_filename = uploaded_file.name
        audio_mimetype = uploaded_file.type

with tab2:
    recorded_audio = st.audio_input("Click the mic to start recording")
    if recorded_audio is not None:
        audio_to_process = recorded_audio.getvalue()
        audio_filename = "recording.wav"
        audio_mimetype = "audio/wav"

st.divider()

# --- Transcribe Button (shared for both inputs) ---
if audio_to_process is not None:
    if st.button("🔍 Transcribe Audio", use_container_width=True):
        with st.spinner("Processing audio... This may take a moment."):
            try:
                files = {"audio_file": (audio_filename, audio_to_process, audio_mimetype)}
                response = requests.post(BACKEND_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success("Transcription Complete!")
                    st.subheader("Repaired Transcript:")
                    st.write(result["text"])
                else:
                    st.error(f"Error from Backend: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is your FastAPI server running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Upload a file or record audio above to get started.")