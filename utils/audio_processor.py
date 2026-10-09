import os

# If you keep a portable ffmpeg inside the project (ffmpeg/bin), use it.
# Otherwise ffmpeg must be installed on your system PATH.
# This must run BEFORE pydub is imported.
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FFMPEG_BIN = os.path.join(_BASE_DIR, "ffmpeg", "bin")
if os.path.isdir(_FFMPEG_BIN):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = os.path.join(_BASE_DIR, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),  # safe file names
        "noplaylist": True,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }
    # Optional: if YouTube blocks you (HTTP 403), set YT_COOKIES_BROWSER=chrome in .env
    browser = os.getenv("YT_COOKIES_BROWSER", "").strip()
    if browser:
        ydl_opts["cookiesfrombrowser"] = (browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        original = ydl.prepare_filename(info)
    return os.path.splitext(original)[0] + ".wav"


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16 kHz mono WAV."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path).set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        audio[start:start + chunk_ms].export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def process_input(source: str) -> list:
    if source.startswith(("http://", "https://")):
        print("Detected URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks
