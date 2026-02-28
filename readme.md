Before running this project, ensure you have the following installed on your system:
run: pip install -r requirements.txt on your terminal

* **Python 3.11**

* **FFmpeg**: Whisper requires the command-line tool `ffmpeg` to be installed on your system to process audio files. 

* Run this on your terminal to check if ffmpeg is installed: ffmpeg -version   *

  * **Windows**: Install via [Scoop](https://scoop.sh/) (`scoop install ffmpeg`) or download the executable and add it to your PATH.
  * **macOS**: `brew install ffmpeg`
  * **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
