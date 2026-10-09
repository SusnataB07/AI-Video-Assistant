# AI Video Assistant

Paste a YouTube link (or a local audio/video file) and get: transcript, title, summary,
action items, key decisions, open questions, and a chat box to ask questions about the video.

## One-time setup (Windows, PowerShell, inside this folder)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r Requirements.txt
```

1. **Add your API key**: copy `.env.example` to `.env`, then fill in the key
   for the provider you use (`LLM_PROVIDER=groq` by default).
2. **ffmpeg** is required. Either:
   - install it (`winget install ffmpeg`) and reopen the terminal, **or**
   - copy a portable `ffmpeg` folder (containing `bin\ffmpeg.exe`) into this project folder.
3. Check ffmpeg works: `ffmpeg -version`

## Run

Command line:
```
python main.py
```
Web UI:
```
streamlit run app.py
```

## Notes
- Test with a short (2-3 minute) video first.
- Switch LLM any time by changing `LLM_PROVIDER` in `.env` (groq / mistral / gemini).
- If YouTube gives HTTP 403: `pip install -U "yt-dlp[default]"`, or add `YT_COOKIES_BROWSER=chrome` to `.env`.
- Rate limit (429) errors: wait 1-2 minutes. The code retries automatically.
- Never upload `.env` to GitHub (it is already in `.gitignore`).
