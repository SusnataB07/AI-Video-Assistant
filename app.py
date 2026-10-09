import html
import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # keep before importing anything from core/

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

/* hide default Streamlit chrome so it feels like a website */
#MainMenu, footer, header[data-testid="stHeader"], .stDeployButton { display: none !important; visibility: hidden; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }

.block-container { max-width: 1120px; padding-top: 1.2rem; padding-bottom: 4rem; }

/* top navigation bar */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 2.2rem;
}
.brand { display: flex; align-items: center; gap: 0.6rem; font-weight: 800; font-size: 1.15rem; color: #0f172a; }
.brand-logo {
    width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center;
    background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #fff; font-size: 1rem;
}
.nav-tags { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.tag {
    font-size: 0.72rem; font-weight: 600; padding: 0.3rem 0.7rem; border-radius: 999px;
    background: #eef2ff; color: #4338ca; border: 1px solid #e0e7ff;
}

/* hero */
.hero { text-align: center; padding: 1.5rem 0 1.6rem; }
.hero h1 {
    font-size: 2.9rem; line-height: 1.15; font-weight: 800; letter-spacing: -0.03em;
    color: #0f172a; margin: 0 0 0.8rem;
}
.hero h1 span {
    background: linear-gradient(135deg, #4f46e5, #9333ea);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { font-size: 1.08rem; color: #475569; max-width: 640px; margin: 0 auto; line-height: 1.6; }

/* the input form as a card */
div[data-testid="stForm"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.3rem 1.4rem 1rem; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}
div[data-testid="stForm"] label p { font-weight: 600; font-size: 0.82rem; color: #334155; }

/* buttons */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button, div.stDownloadButton > button {
    width: 100%; border-radius: 10px; font-weight: 600; padding: 0.55rem 1rem; border: 1px solid #e2e8f0;
}
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #fff; border: none;
}
div[data-testid="stFormSubmitButton"] > button:hover { filter: brightness(1.08); color: #fff; }

/* feature cards on the landing page */
.feature {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.3rem 1.2rem; height: 100%;
}
.feature .icon {
    width: 40px; height: 40px; border-radius: 10px; background: #eef2ff; display: grid;
    place-items: center; font-size: 1.2rem; margin-bottom: 0.8rem;
}
.feature h4 { margin: 0 0 0.35rem; font-size: 1rem; color: #0f172a; }
.feature p { margin: 0; font-size: 0.88rem; color: #64748b; line-height: 1.55; }
.section-label {
    text-align: center; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6366f1; margin: 2.4rem 0 1rem;
}

/* result header */
.result-head {
    background: linear-gradient(135deg, #eef2ff, #f5f3ff); border: 1px solid #e0e7ff;
    border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.result-head .eyebrow { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #6366f1; }
.result-head h2 { margin: 0.3rem 0 0; font-size: 1.7rem; letter-spacing: -0.02em; color: #0f172a; }

/* metrics */
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 0.8rem 1rem;
}

/* section navigation (radio styled like pills) */
div[role="radiogroup"] { gap: 0.4rem; }
div[role="radiogroup"] > label {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 999px; padding: 0.35rem 1rem;
}

/* bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; background: #ffffff; }

.footer { text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ──────────────────────────────────────────────────────────────
st.session_state.setdefault("result", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("view", "Overview")
st.session_state.setdefault("last_error", None)


def build_report(r: dict) -> str:
    """Plain-text/markdown report the user can download."""
    return (
        f"# {r['title']}\n\n"
        f"## Summary\n{r['summary']}\n\n"
        f"## Action Items\n{r['action_items']}\n\n"
        f"## Key Decisions\n{r['key_decisions']}\n\n"
        f"## Open Questions\n{r['open_questions']}\n\n"
        f"## Full Transcript\n{r['transcript']}\n"
    )


# ─── Navigation bar ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="brand"><div class="brand-logo">🎬</div> AI Video Assistant</div>
  <div class="nav-tags">
    <span class="tag">Whisper</span><span class="tag">LangChain</span>
    <span class="tag">ChromaDB</span><span class="tag">RAG Chat</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Hero ───────────────────────────────────────────────────────────────────────
if not st.session_state.result:
    st.markdown("""
    <div class="hero">
      <h1>Turn any video into <span>clear insights</span></h1>
      <p>Paste a YouTube link, or upload an audio or video file. Get a transcript, a summary, action items
         and decisions, then chat with the video to find any detail.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)

# ─── Input form ─────────────────────────────────────────────────────────────────
with st.form("analyse_form", border=True):
    c1, c2, c3 = st.columns([5, 1.6, 1.6], gap="medium", vertical_alignment="bottom")
    with c1:
        source = st.text_input("YouTube URL or file path", placeholder="https://www.youtube.com/watch?v=...")
    with c2:
        language = st.selectbox("Language", ["english", "hinglish"], index=0)
    with c3:
        run_btn = st.form_submit_button("Analyse  →")
    uploaded = st.file_uploader(
        "…or upload an audio / video file from your computer",
        type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "ogg", "flac"],
    )

# ─── Run the pipeline ───────────────────────────────────────────────────────────
if run_btn:
    if not source.strip() and uploaded is None:
        st.warning("Please enter a YouTube URL, a file path, or upload a file.")
    else:
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.last_error = None
        finished = False

        with st.status("Analysing your video…", expanded=True) as status:
            try:
                st.write("🔊 Preparing audio…")
                if uploaded is not None:
                    os.makedirs("downloads", exist_ok=True)
                    source_path = os.path.join("downloads", "upload_" + uploaded.name)
                    with open(source_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                else:
                    source_path = source.strip().strip('"')
                chunks = process_input(source_path)

                st.write("📝 Transcribing speech (this is the slowest step)…")
                transcript = transcribe_all(chunks, language)

                st.write("🏷️ Writing title and summary…")
                title = generate_title(transcript)
                summary = summarize(transcript)

                st.write("🔍 Extracting action items, decisions and questions…")
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

                st.write("🧠 Building the chat index…")
                rag_chain = build_rag_chain(transcript)

                st.session_state.result = {
                    "title": title,
                    "transcript": transcript,
                    "summary": summary,
                    "action_items": action_items,
                    "key_decisions": decisions,
                    "open_questions": questions,
                    "rag_chain": rag_chain,
                }
                st.session_state.view = "Overview"
                status.update(label="Analysis complete", state="complete", expanded=False)
                finished = True
            except Exception as e:  # show the problem instead of a crash page
                st.session_state.last_error = str(e)
                status.update(label="Something went wrong", state="error", expanded=True)

        if finished:
            time.sleep(0.3)
            st.rerun()

if st.session_state.last_error:
    st.error(f"Error: {st.session_state.last_error}")

# ─── Landing page (no result yet) ───────────────────────────────────────────────
r = st.session_state.result

if not r:
    st.markdown('<div class="section-label">What you get</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    cards = [
        ("📝", "Accurate transcript", "Speech is converted to text locally with Whisper, split into chunks so long videos work."),
        ("📋", "Instant summary", "A short title, a bullet summary, action items, key decisions and open questions."),
        ("💬", "Chat with the video", "Ask a question in plain language and get an answer grounded in the transcript."),
    ]
    for col, (icon, head, body) in zip((f1, f2, f3), cards):
        with col:
            st.markdown(
                f'<div class="feature"><div class="icon">{icon}</div><h4>{head}</h4><p>{body}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown('<div class="footer">Built with Python · Whisper · LangChain · ChromaDB · Streamlit</div>', unsafe_allow_html=True)
    st.stop()

# ─── Results ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="result-head"><div class="eyebrow">Session title</div>'
    f'<h2>{html.escape(r["title"])}</h2></div>',
    unsafe_allow_html=True,
)

words = len(r["transcript"].split())
m1, m2, m3, m4 = st.columns([1, 1, 1, 1.4], gap="medium", vertical_alignment="center")
m1.metric("Transcript words", f"{words:,}")
m2.metric("Reading time", f"{max(1, round(words / 200))} min")
m3.metric("Chat questions", len(st.session_state.chat_history) // 2)
with m4:
    st.download_button(
        "⬇  Download report",
        data=build_report(r),
        file_name="video-report.md",
        mime="text/markdown",
    )

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

view = st.radio(
    "Section",
    ["Overview", "Insights", "Transcript", "Chat"],
    horizontal=True,
    label_visibility="collapsed",
    key="view",
)

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

if view == "Overview":
    with st.container(border=True):
        st.subheader("📋 Summary")
        st.markdown(r["summary"])

elif view == "Insights":
    a, b, c = st.columns(3, gap="medium")
    with a:
        with st.container(border=True):
            st.subheader("✅ Action items")
            st.markdown(r["action_items"])
    with b:
        with st.container(border=True):
            st.subheader("🔑 Key decisions")
            st.markdown(r["key_decisions"])
    with c:
        with st.container(border=True):
            st.subheader("❓ Open questions")
            st.markdown(r["open_questions"])

elif view == "Transcript":
    with st.container(border=True):
        st.subheader("📝 Full transcript")
        st.text_area("Transcript", r["transcript"], height=420, label_visibility="collapsed")

else:  # Chat
    with st.container(border=True):
        st.subheader("💬 Chat with your video")
        st.caption("Answers come only from this video's transcript.")

        if not st.session_state.chat_history:
            st.info("Try: “What are the main points?” or “Who is mentioned in the video?”")

        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    prompt = st.chat_input("Ask anything about this video…")
    if prompt and prompt.strip():
        question = prompt.strip()
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = ask_question(r["rag_chain"], question)
                except Exception as e:
                    answer = f"Sorry, I could not get an answer: {e}"
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

st.markdown('<div class="footer">Built with Python · Whisper · LangChain · ChromaDB · Streamlit</div>', unsafe_allow_html=True)