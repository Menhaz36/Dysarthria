import streamlit as st
import requests

# Set page config
st.set_page_config(page_title="Dysarthria ASR Interface", page_icon="🎙️")

st.title("🎙️ Whisper ASR for Dysarthria Research")
st.markdown("""
Upload an audio file (WAV, MP3, or M4A) to transcribe it using the locally hosted Whisper model.
""")

# URL of your FastAPI backend
BACKEND_URL = "http://127.0.0.1:8000/transcribe"

# 1. File Uploader
uploaded_file = st.file_uploader("Choose an audio file, make sure the file start with audio.*", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # 2. Audio Preview
    st.audio(uploaded_file, format='audio/wav')
    
    # 3. Transcription Button
    if st.button("Transcribe Audio"):
        with st.spinner("Processing audio... This may take a moment."):
            try:
                # Prepare the file for the POST request
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send request to FastAPI
                response = requests.post(BACKEND_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Transcription Complete!")
                    
                    # Display Results
                    st.subheader("Resulting Text:")
                    st.write(result["text"])
                    
                    # Display Metadata
                    st.info(f"Detected Language: {result.get('language', 'Unknown')}")
                else:
                    st.error(f"Error from Backend: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is your FastAPI server running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.divider()
