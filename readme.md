Before running this project, ensure you have the following installed on your system:
run: pip install -r requirements.txt on your terminal

* **Python 3.11**

* **FFmpeg**: Whisper requires the command-line tool `ffmpeg` to be installed on your system to process audio files. 
*Go to [this page](https://www.gyan.dev/ffmpeg/builds/)
*Scroll down to the "release builds" section and click the link for ffmpeg-release-full-shared.7z *
*extract it , open the bin folder and  copy the path *
*Press your Windows Key, type Environment Variables, and hit Enter*
*In the new window, look at the bottom list called System variables. Scroll down until you find the variable named Path, select it, and click Edit*
*Click the New button on the right side.*
*Type or paste the exact path to your new bin folder eg:D:\ffmpeg\bin*
*Click OK and restart your device.*



Backend: uvicorn main1:app --reload
Frontend: streamlit run Frontend.py
