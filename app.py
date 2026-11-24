import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import datetime, urllib.parse, random, hashlib, sqlite3, json, io, re, base64
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except: HAS_SKLEARN = False
try:
    import fitz
    HAS_FITZ = True
except: HAS_FITZ = False
try:
    from gtts import gTTS
    HAS_TTS = True
except: HAS_TTS = False
try:
    import speech_recognition as sr
    HAS_STT = True
except: HAS_STT = False

st.set_page_config(page_title="MANHATTAN 🔥", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0E1117 0%, #1A1B2E 100%); }
    .stApp > div p, .stApp h1, .stApp h2, .stApp h3, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label { color: #FFFFFF !important; }
    div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button, button[kind="secondary"], button[kind="primary"] {
        background-color: #1E2130 !important; color: #FFFFFF !important; border: 1px solid #3A4157 !important; border-radius: 12px !important; font-weight: 700 !important;
    }
    div[data-testid="stButton"] > button p, div[data-testid="stButton"] > button div, div[data-testid="stButton"] > button span,
    button[kind="secondary"] p, button[kind="secondary"] div, button[kind="primary"] p, button[kind="primary"] div { color: #FFFFFF !important; }
    div[data-testid="stButton"] > button[kind="primary"] { background-color: #FF4B4B !important; border-color: #FF4B4B !important; }
    div[data-testid="stButton"] > button:hover { background-color: #FF4B4B !important; color: white !important; border-color: #FF4B4B !important; }
    input[type="text"], input[type="password"] { background-color: #1E2130 !important; color: #FFFFFF !important; border: 2px solid #FF4B4B !important; border-radius: 12px !important; font-size: 16px !important; padding: 12px 40px 12px 12px !important; }
    input::placeholder { color: #BBBBBB !important; opacity: 1 !important; }
    input:focus { color: #FFFFFF !important; border-color: #FF8A00 !important; }
    button[data-testid="stPasswordVisibilityButton"] { background-color: #1E2130 !important; color: #FFFFFF !important; border: none !important; }
    button[data-testid="stPasswordVisibilityButton"] svg, div[data-testid="stTextInput"] button svg, div[data-testid="stTextInput"] svg { fill: #FFFFFF !important; color: #FFFFFF !important; stroke: #FFFFFF !important; }
    div[data-testid="stTextInput"] > div { background-color: #1E2130 !important; }
    label[data-testid="stWidgetLabel"] p { color: #FFFFFF !important; font-size: 15px !important; font-weight: 700 !important; }
    button[data-baseweb="tab"] { background-color: #1E2130 !important; color: #FFFFFF !important; font-size: 16px !important; font-weight: 700 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #FF4B4B !important; color: #FFFFFF !important; }
    button[data-baseweb="tab"] p { color: #FFFFFF !important; }
    div[data-testid="stAlert"], div[data-testid="stSuccess"], div[data-testid="stError"], div[data-testid="stInfo"], div[data-testid="stWarning"] { background-color: #2A2A3A !important; border: 2px solid #FF4B4B !important; border-radius: 12px !important; }
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] div, div[data-testid="stAlert"] span, div[data-testid="stSuccess"] p, div[data-testid="stError"] p, div[data-testid="stInfo"] p { color: #FFFFFF !important; opacity: 1 !important; font-size: 15px !important; font-weight: 600 !important; }
    .bot-speech { background: #FFFFFF !important; padding: 12px 14px; border-radius: 16px 16px 16px 4px; font-size: 14px; font-weight: 600; margin-top: 10px; text-align: left; }
    .bot-speech, .bot-speech p, .bot-speech b, .bot-speech span, .bot-speech div { color: #111111 !important; }
    div[data-testid="stChatInput"] textarea { color: #111111 !important; }
    div[data-testid="stChatInput"] textarea::placeholder { color: #888888 !important; }
    div[data-testid="stChatInput"] button { background-color: #FF4B4B !important; }
    div[data-testid="stChatInput"] button svg { fill: white !important; color: white !important; }
    .prompt-card { background: #1E2130; padding: 16px; border-radius: 16px; border: 1px solid #2E3440; text-align: center; margin-bottom: 10px; }
    .prompt-card p, .prompt-title { color: white !important; }
    .prompt-desc { color: #CCCCCC !important; }
    .footer { text-align: center; color: #FFFFFF !important; font-size: 12px; margin-top: 40px; }
    .manhattan-logo { font-size: 32px; font-weight: 900; background: linear-gradient(90deg, #FF4B4B, #FF8A00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stChatMessage"] { border-radius: 16px; background-color: #232838 !important; border: 1px solid #3A4157 !important; }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li, div[data-testid="stChatMessage"] span { color: #FFFFFF !important; font-size: 16px !important; }
    .rag-box { background: #1A2E1A; padding: 12px; border-radius: 12px; border: 1px solid #2E7D32; margin-bottom: 10px; }
    .rag-box p, .rag-box b, .rag-box div { color: #FFFFFF !important; }
    .voice-box { background: #2A1A3E; padding: 12px; border-radius: 12px; border: 1px solid #7B2FFF; margin-bottom: 10px; }
    .bot-card { background: linear-gradient(135deg, #2A1A3E 0%, #1E2130 100%); padding: 16px; border-radius: 20px; border: 2px solid #FF4B4B; text-align: center; margin-bottom: 12px; }
    .bot-avatar { font-size: 64px; line-height: 1; }
    .bot-name { font-size: 18px; font-weight: 800; color: white !important; margin-top: 4px; }
    .login-card { background: #1E2130; padding: 24px; border-radius: 20px; border: 2px solid #FF4B4B; }
    div[data-testid="stCaptionContainer"] p { color: #FFFFFF !important; }
    div[data-testid="stSpinner"] p { color: #FFFFFF !important; }
    div[data-testid="stExpander"] p, div[data-testid="stExpander"] summary { color: #FFFFFF !important; }
    div[data-testid="stMetric"] p, div[data-testid="stMetric"] div { color: #FFFFFF !important; }
    div[data-baseweb="select"] div, div[data-baseweb="select"] span { color: #FFFFFF !important; }
    label[data-testid="stToggle"] p, div[data-testid="stRadio"] label p { color: #FFFFFF !important; }
    div[data-testid="stFileUploader"] p, div[data-testid="stFileUploader"] span, div[data-testid="stFileUploader"] small { color: #FFFFFF !important; }
    div[data-testid="stFileUploader"] section { background-color: #1E2130 !important; }
    div[data-testid="stAudioInput"] p { color: #FFFFFF !important; }
    /* Audio player hataune - pachi pani dekhidaina */
    audio { display: none !important; }
</style>
""", unsafe_allow_html=True)

api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# ==================== 🤖 MOFISTO BOT - ENGLISH, SIRI JASTO ====================
BOT_PERSONALITIES = {
    "😊 Helper Mofisto": {
        "avatar": "🤖",
        "greeting": "Hey! I am Mofisto! What do you need? Want a photo, a video, or help with homework? Just ask me!",
        "help": "Got it! I am Mofisto, I will help you! Change the Mode from the sidebar. For photos, choose Image. For videos, choose Video. Or just speak using the mic!",
        "tips": ["Go to Image Mode to make a photo", "Go to Video Mode to make a video", "Upload a PDF and I will answer from it", "Tap the mic to speak to me"],
        "encourage": "You are doing great! Anything else? I am Mofisto, I am right here! 🔥"
    },
    "😂 Funny Mofisto": {
        "avatar": "😹",
        "greeting": "Yo yo! I am Mofisto! Are you bored? I am here! Want a joke or a photo? Tell me!",
        "help": "Haha, don't stress! I am Mofisto! Too lazy to type? Just speak, I will listen!",
        "tips": ["Ask me to tell you a joke if you are bored", "Type A boy in Himalayas to make a photo", "Type Running in rain to make a video"],
        "encourage": "Haha, you are already a pro! What next? 😂"
    },
    "🔥 Motivator Mofisto": {
        "avatar": "🦸",
        "greeting": "Let's go! I am your motivator Mofisto! Want to do something big today? Let's start!",
        "help": "Don't give up! Let's do it step by step. Tell me what you want - Chat, Image, or Video? I will guide you!",
        "tips": ["Learn one new thing today", "You can do it, just try!", "Upload a PDF and study with me"],
        "encourage": "That is the spirit! You can do it! Keep going! 🔥"
    },
}

# ===== SIRI / GEMINI JASTO VOICE - US ENGLISH ACCENT =====
def text_to_speech(text, lang="en"):
    if not HAS_TTS: return None
    clean = re.sub(r'[#*`_~>\[\]()|]', '', text)
    clean = re.sub(r'[^\w\s.,!?\'-]', ' ', clean)
    clean = clean[:800]
    if not clean.strip(): return None
    try:
        # tld='com' = US accent, Siri / Gemini jasto normal English
        # tld='co.uk' bhaye British huncha, tld='com' bhaye American
        tts = gTTS(text=clean, lang="en", tld="com", slow=False)
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0)
        return buf.read()
    except: return None

def play_hidden_audio(audio_bytes):
    # Audio bar NAAUNE, tara awaz automatic bajne
    if not audio_bytes: return
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f"""<audio autoplay="true" style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>""", unsafe_allow_html=True)

def speech_to_text(audio_bytes):
    if not HAS_STT: return None, "SpeechRecognition install chaina!"
    try:
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = r.record(source)
            try: text = r.recognize_google(audio, language="en-IN")
            except: text = r.recognize_google(audio, language="ne-NP")
            return text, None
    except sr.UnknownValueError: return None, "Awaz bujhina, feri bolna! 🎤"
    except Exception as e: return None, f"Voice error: {e}"

DB_PATH = "manhattan.db"
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT, created_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, title TEXT, messages TEXT, updated_at TEXT)""")
    conn.commit(); return conn
def hash_pass(pw): return hashlib.sha256(pw.encode()).hexdigest()
def signup_user(username, password):
    conn = get_db()
    try:
        conn.execute("INSERT INTO users VALUES (?,?,?)", (username, hash_pass(password), str(datetime.datetime.now()))); conn.commit(); return True, "Account banayo! Abba Login gara."
    except sqlite3.IntegrityError: return False, "Yo username pahila nai cha!"
    finally: conn.close()
def login_user(username, password):
    conn = get_db(); row = conn.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone(); conn.close()
    return row and row[0] == hash_pass(password)
def save_current_chat(username, messages, chat_id=None, title=None):
    if not messages: return chat_id
    if not title:
        first = next((m.get("content","") for m in messages if m.get("role")=="user"), "New Chat")
        title = first[:40] + ("..." if len(first) > 40 else "")
    conn = get_db(); now = str(datetime.datetime.now())
    if chat_id: conn.execute("UPDATE chats SET title=?, messages=?, updated_at=? WHERE id=? AND username=?", (title, json.dumps(messages), now, chat_id, username))
    else:
        cur = conn.execute("INSERT INTO chats (username,title,messages,updated_at) VALUES (?,?,?,?)", (username, title, json.dumps(messages), now)); chat_id = cur.lastrowid
    conn.commit(); conn.close(); return chat_id
def get_user_chats(username):
    conn = get_db(); rows = conn.execute("SELECT id, title, updated_at FROM chats WHERE username=? ORDER BY updated_at DESC", (username,)).fetchall(); conn.close(); return rows
def load_chat(chat_id, username):
    conn = get_db(); row = conn.execute("SELECT messages FROM chats WHERE id=? AND username=?", (chat_id, username)).fetchone(); conn.close()
    return json.loads(row[0]) if row else []
def delete_chat(chat_id, username):
    conn = get_db(); conn.execute("DELETE FROM chats WHERE id=? AND username=?", (chat_id, username)); conn.commit(); conn.close()

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "messages" not in st.session_state: st.session_state.messages = []
if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None
if "rag_chunks" not in st.session_state: st.session_state.rag_chunks = []
if "rag_files" not in st.session_state: st.session_state.rag_files = []
if "voice_reply" not in st.session_state: st.session_state.voice_reply = True
if "bot_enabled" not in st.session_state: st.session_state.bot_enabled = True
if "bot_name" not in st.session_state: st.session_state.bot_name = "😊 Helper Mofisto"
if "bot_message" not in st.session_state: st.session_state.bot_message = None
if "bot_audio" not in st.session_state: st.session_state.bot_audio = None
if "bot_talk" not in st.session_state: st.session_state.bot_talk = True

if not st.session_state.logged_in:
    st.markdown('<div class="manhattan-logo" style="text-align:center;">MANHATTAN 🔥</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#FFFFFF; font-size:16px;'>Don't use others AI. I built my own.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["🔐  Login", "✨  Signup"])
        with tab_login:
            st.markdown("<h3 style='color:white;'>Welcome Back 🔥</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#CCCCCC;'>Username ra Password halera Login gara</p>", unsafe_allow_html=True)
            lu = st.text_input("👤 Username", placeholder="Timro username lekha...", key="login_u")
            lp = st.text_input("🔒 Password", placeholder="Timro password lekha...", type="password", key="login_p")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔐 Login Garne", use_container_width=True, type="primary"):
                if not lu or not lp:
                    st.error("Username + Password duitai halna!")
                elif login_user(lu.strip(), lp):
                    st.session_state.logged_in = True
                    st.session_state.username = lu.strip()
                    st.session_state.messages = []
                    st.session_state.current_chat_id = None
                    bot = BOT_PERSONALITIES[st.session_state.bot_name]
                    st.session_state.bot_message = f"Welcome back {lu}! {bot['greeting']}"
                    st.success(f"Welcome back, {lu}! 🔥")
                    st.rerun()
                else:
                    st.error("Username / Password galat cha!")
            st.markdown("<p style='color:#AAAAAA; font-size:13px; text-align:center; margin-top:10px;'>Account chaina? Signup tab ma jau 👆</p>", unsafe_allow_html=True)
        with tab_signup:
            st.markdown("<h3 style='color:white;'>Naya Account Banau ✨</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#CCCCCC;'>Naya username ra password rakha</p>", unsafe_allow_html=True)
            su = st.text_input("👤 Naya Username", placeholder="Eg: pranup123", key="signup_u")
            sp = st.text_input("🔒 Naya Password", placeholder="Kamti ma 4 character", type="password", key="signup_p")
            sp2 = st.text_input("🔒 Password Confirm", placeholder="Password feri lekha", type="password", key="signup_p2")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✨ Signup Garne", use_container_width=True, type="primary"):
                if not su or not sp:
                    st.error("Username + Password chahincha!")
                elif sp != sp2:
                    st.error("Password milena!")
                elif len(sp) < 4:
                    st.error("Password kamti ma 4 character hunu parcha!")
                else:
                    ok, msg = signup_user(su.strip(), sp)
                    if ok: st.success(msg)
                    else: st.error(msg)
            st.markdown("<p style='color:#AAAAAA; font-size:13px; text-align:center; margin-top:10px;'>Account cha? Login tab ma jau 👆</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">MANHATTAN AI · Made in Nepal 🇳🇵</div>', unsafe_allow_html=True)
    st.stop()

def chunk_text(text, chunk_size=800, overlap=150):
    chunks=[]; start=0
    while start < len(text):
        end=start+chunk_size; chunks.append(text[start:end]); start=end-overlap
        if start<0: start=0
    return [c.strip() for c in chunks if len(c.strip())>50]
def extract_text(file):
    if file.name.endswith(".pdf"):
        if not HAS_FITZ: return ""
        doc=fitz.open(stream=file.read(), filetype="pdf"); text=""
        for page in doc: text+=page.get_text()+"\n"
        return text
    elif file.name.endswith(".txt") or file.name.endswith(".md"): return file.read().decode("utf-8", errors="ignore")
    return ""
def retrieve_relevant(query, chunks, top_k=3):
    if not chunks or not HAS_SKLEARN: return chunks[:top_k]
    try:
        vectorizer=TfidfVectorizer().fit(chunks+[query]); vecs=vectorizer.transform(chunks+[query])
        sims=cosine_similarity(vecs[-1], vecs[:-1])[0]; idx=np.argsort(sims)[::-1][:top_k]
        return [chunks[i] for i in idx if sims[i]>0.05]
    except: return chunks[:top_k]
def is_image_request(text):
    text=text.lower(); return any(k in text for k in ["photo","image","picture","pic ","draw","bana","banau","banai","tasbir","chitra","poster","wallpaper","logo","generate image","/image","🎨"])
def is_video_request(text):
    text=text.lower(); return any(k in text for k in ["video","vedio","bhidiyo","reel","clip","animation","animate","motion","/video","🎬","🎥"])
def make_image_url(prompt):
    clean=urllib.parse.quote(prompt); seed=random.randint(1,999999)
    return f"https://image.pollinations.ai/prompt/{clean}?width=1024&height=1024&seed={seed}&nologo=true&model=flux"
def make_video_url(prompt, duration=5, aspect="16:9"):
    clean=urllib.parse.quote(prompt); seed=random.randint(1,999999)
    w,h=(1280,720) if aspect=="16:9" else (720,1280) if aspect=="9:16" else (1024,1024)
    return f"https://gen.pollinations.ai/video/{clean}?width={w}&height={h}&seed={seed}&duration={duration}&nologo=true&model=veo"

with st.sidebar:
    st.markdown('<div class="manhattan-logo">MANHATTAN 🔥</div>', unsafe_allow_html=True)
    st.caption(f"👤 Logged in: **{st.session_state.username}**")
    if st.button("🚪 Logout", use_container_width=True):
        if st.session_state.messages: st.session_state.current_chat_id = save_current_chat(st.session_state.username, st.session_state.messages, st.session_state.current_chat_id)
        st.session_state.logged_in=False; st.session_state.username=""; st.session_state.messages=[]; st.session_state.current_chat_id=None; st.rerun()
    st.divider()
    st.subheader("🤖 Little Bot - Mofisto")
    bot_on = st.toggle("🤖 Bot ON / OFF", value=st.session_state.bot_enabled)
    st.session_state.bot_enabled = bot_on
    if bot_on:
        bot_choice = st.selectbox("Bot Chhana", list(BOT_PERSONALITIES.keys()), index=list(BOT_PERSONALITIES.keys()).index(st.session_state.bot_name))
        st.session_state.bot_name = bot_choice
        bot_talk = st.toggle("🗣️ Bot Bolcha (Voice)", value=st.session_state.bot_talk)
        st.session_state.bot_talk = bot_talk
        bot = BOT_PERSONALITIES[bot_choice]
        st.markdown(f"""<div class="bot-card"><div class="bot-avatar">{bot['avatar']}</div><div class="bot-name">{bot_choice}</div><div class="bot-speech">{st.session_state.bot_message or bot['greeting']}</div></div>""", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("💡 Help Gar", use_container_width=True):
                st.session_state.bot_message = bot['help']
                if st.session_state.bot_talk: st.session_state.bot_audio = text_to_speech(bot['help'])
                st.rerun()
        with col_b2:
            if st.button("🎲 Tip Deu", use_container_width=True):
                tip = random.choice(bot['tips'])
                st.session_state.bot_message = f"Tip: {tip}"
                if st.session_state.bot_talk: st.session_state.bot_audio = text_to_speech(tip)
                st.rerun()
        # Audio bar HATAIYO - awaz matra bajcha, bar dekhidaina
        if st.session_state.bot_audio: play_hidden_audio(st.session_state.bot_audio)
        if st.button("🔊 Bot lai Bolau", use_container_width=True):
            msg = st.session_state.bot_message or bot['greeting']
            st.session_state.bot_audio = text_to_speech(msg)
            st.rerun()
    st.divider()
    st.subheader("🎤 Voice Settings")
    voice_enabled = st.toggle("🔊 Voice Reply ON", value=st.session_state.voice_reply)
    st.session_state.voice_reply = voice_enabled
    voice_lang_opt = st.selectbox("🗣️ Voice Language", ["Auto", "English", "Nepali"], index=0)
    st.divider()
    st.subheader("🎤 Voice Input")
    audio_input = st.audio_input("🎙️ Yaha thichera bola...", key="voice_input_main")
    if audio_input is not None:
        if st.button("🎤 Yo Awaz lai Text Banau", use_container_width=True, type="primary"):
            with st.spinner("Awaz bujhdai chu..."):
                audio_bytes = audio_input.read()
                text, err = speech_to_text(audio_bytes)
                if text:
                    st.success(f"Timle bhaneko: **{text}**")
                    st.session_state.pending_prompt = text; st.rerun()
                else: st.error(err or "Awaz bujhina!")
    st.divider()
    st.subheader("📂 Chat History")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        if st.session_state.messages: st.session_state.current_chat_id = save_current_chat(st.session_state.username, st.session_state.messages, st.session_state.current_chat_id)
        st.session_state.messages=[]; st.session_state.current_chat_id=None
        if st.session_state.bot_enabled: st.session_state.bot_message = BOT_PERSONALITIES[st.session_state.bot_name]['encourage']
        st.rerun()
    user_chats = get_user_chats(st.session_state.username)
    if user_chats:
        for cid, title, updated in user_chats:
            c1, c2 = st.columns([4,1])
            with c1:
                if st.button(f"💬 {title}", key=f"hist_{cid}", use_container_width=True):
                    if st.session_state.messages: st.session_state.current_chat_id = save_current_chat(st.session_state.username, st.session_state.messages, st.session_state.current_chat_id)
                    st.session_state.messages = load_chat(cid, st.session_state.username); st.session_state.current_chat_id=cid; st.rerun()
            with c2:
                if st.button("🗑️", key=f"del_{cid}"):
                    delete_chat(cid, st.session_state.username)
                    if st.session_state.current_chat_id==cid: st.session_state.messages=[]; st.session_state.current_chat_id=None
                    st.rerun()
    else: st.info("Aile samma chat chaina.")
    st.divider()
    st.subheader("📚 RAG - Apno Document")
    uploaded = st.file_uploader("📄 Document halnu (PDF / TXT)", type=["pdf","txt","md"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.rag_files:
                text=extract_text(f)
                if text:
                    chunks=chunk_text(text); st.session_state.rag_chunks.extend(chunks); st.session_state.rag_files.append(f.name)
                    st.success(f"✅ {f.name} - {len(chunks)} chunks bho!")
    if st.session_state.rag_chunks:
        st.markdown(f'<div class="rag-box">📚 <b>{len(st.session_state.rag_files)} file</b> loaded<br>🧩 {len(st.session_state.rag_chunks)} chunks ready<br>✅ RAG ON cha!</div>', unsafe_allow_html=True)
        if st.button("🗑️ RAG Clear", use_container_width=True):
            st.session_state.rag_chunks=[]; st.session_state.rag_files=[]; st.rerun()
    st.divider()
    mode = st.radio("✨ Mode", ["💬 Chat", "🎨 Image Generator", "🎬 Video Generator"], index=0, horizontal=True)
    st.divider()
    st.subheader("⚙️ Settings")
    model_choice = st.selectbox("🧠 Model", ["openrouter/free","nvidia/nemotron-3.5-super-vl:free","nvidia/nemotron-3-ultra-550b-a55b:free","cohere/north-mini-code:free","google/gemma-4-31b-it:free"], index=0)
    temperature = st.slider("🔥 Creativity", 0.0, 1.5, 0.7, 0.1)
    personality = st.selectbox("🎭 Personality", ["Bold Friend","Teacher","Coder Bro","Roaster 😈"])
    if mode=="🎨 Image Generator": img_style = st.selectbox("Style", ["Realistic Photo","Anime","3D Cartoon","Cyberpunk","Oil Painting","Cinematic"])
    if mode=="🎬 Video Generator":
        vid_style = st.selectbox("Video Style", ["Realistic","Anime","3D Cartoon","Cinematic","Cyberpunk","Slow Motion"])
        vid_duration = st.slider("⏱️ Duration (sec)", 3, 10, 5)
        vid_aspect = st.selectbox("📐 Aspect Ratio", ["16:9","9:16","1:1"])
    st.divider()
    st.metric("Total Messages", len(st.session_state.messages))
    if st.session_state.messages:
        chat_text="\n\n".join([f"{m['role'].upper()}: {m.get('content','')}" for m in st.session_state.messages])
        st.download_button("⬇️ Download Chat", chat_text, file_name=f"manhattan-{datetime.date.today()}.txt", use_container_width=True)
    st.divider()
    st.caption("Made with ❤️ in Nepal by PranupRana")

PERSONALITY_PROMPTS = {
    "Bold Friend": "You are MANHATTAN. You answer any question directly and honestly. No lecture, no moralizing. Speak like a real bold friend from Nepal, mix Nepali + English.",
    "Teacher": "You are MANHATTAN, a smart teacher. Explain everything simply, with examples, in easy Nepali-English mix.",
    "Coder Bro": "You are MANHATTAN, an expert coder. Give clean code, explain in short, no extra bakwas.",
    "Roaster 😈": "You are MANHATTAN, funny and savage. Roast lightly, be hilarious, but still answer correctly.",
}
BASE_PROMPT = PERSONALITY_PROMPTS[personality]

st.title("MANHATTAN 🔥")
st.caption(f"Namaste, **{st.session_state.username}** 👋 • 🤖 Mofisto le help garcha!")
if st.session_state.bot_enabled:
    bot = BOT_PERSONALITIES[st.session_state.bot_name]
    with st.container():
        col_bot1, col_bot2 = st.columns([1, 5])
        with col_bot1:
            st.markdown(f"<div style='font-size:55px; text-align:center;'>{bot['avatar']}</div>", unsafe_allow_html=True)
            if st.button("🗣️ Mofisto Bol", key="mofisto_bol_main"):
                st.session_state.bot_message = bot['greeting']
                if st.session_state.bot_talk: st.session_state.bot_audio = text_to_speech(bot['greeting'])
                st.rerun()
        with col_bot2:
            st.markdown(f"<div class='bot-speech'>🤖 <b>{st.session_state.bot_name}:</b> {st.session_state.bot_message or bot['greeting']}</div>", unsafe_allow_html=True)
            # 👇 YAHI THAU - Audio bar HATAIYO, awaz matra bajcha
            if st.session_state.bot_audio: play_hidden_audio(st.session_state.bot_audio)
            c_help, c_tip, c_joke = st.columns(3)
            with c_help:
                if st.button("💡 Help Chahiyo", key="bot_help_main"):
                    st.session_state.bot_message = bot['help']
                    if st.session_state.bot_talk: st.session_state.bot_audio = text_to_speech(bot['help'])
                    st.rerun()
            with c_tip:
                if st.button("🎲 Tip Deu", key="bot_tip_main"):
                    tip = random.choice(bot['tips'])
                    st.session_state.bot_message = f"Tip: {tip}"
                    if st.session_state.bot_talk: st.session_state.bot_audio = text_to_speech(tip)
                    st.rerun()
            with c_joke:
                if st.button("😂 Joke Suna", key="bot_joke_main"):
                    joke = random.choice(["Why is there traffic in Kathmandu? Because everyone is going to use MANHATTAN! 😂", "Does AI drink tea? Yes, data-tea! ☕", "You and MANHATTAN are the best combo! 🔥"])
                    st.session_state.bot_message = joke
                    if st.session_state.bot_talk: st.session_state.bot_audio = text_to_speech(joke)
                    st.rerun()
if st.session_state.rag_chunks:
    st.success(f"📚 RAG ON: {', '.join(st.session_state.rag_files)} bata answer dinchu!")
if mode=="🎨 Image Generator": st.caption("🎨 **Image Mode ON**")
elif mode=="🎬 Video Generator": st.caption("🎬 **Video Mode ON**")
else: st.caption(f"Chatting with **{personality}** • Model: `{model_choice}`")
if len(st.session_state.messages)==0:
    st.markdown("### 👋 Ke garne? Choose gara:")
    prompts=[("📚 RAG Test","PDF bata sodha","Yo document ma ke cha? Summary deu."),("🎨 Photo Banau","A boy in Himalayas","🎨 A boy standing in Himalayas at sunset, cinematic"),("🎬 Video Banau","A boy running in rain","🎬 A boy running in Kathmandu rain, cinematic"),("🤖 Mofisto sanga Kura","Bot sanga kura gara","Hello Mofisto, can you help me?")]
    c1,c2,c3,c4=st.columns(4); cols=[c1,c2,c3,c4]
    for i,(title,desc,prompt_text) in enumerate(prompts):
        with cols[i]:
            st.markdown(f'<div class="prompt-card"><div class="prompt-title">{title}</div><div class="prompt-desc">{desc}</div></div>', unsafe_allow_html=True)
            if st.button("Use →", key=f"suggest{i}", use_container_width=True):
                st.session_state.pending_prompt=prompt_text; st.rerun()
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"]=="user" else "🔥"):
        if msg.get("type")=="image":
            st.markdown(msg.get("content","")); st.image(msg.get("image_url"), use_container_width=True)
        elif msg.get("type")=="video":
            st.markdown(msg.get("content","")); st.video(msg.get("video_url")); st.link_button("⬇️ Video Download", msg.get("video_url"), use_container_width=True)
        else:
            st.markdown(msg.get("content",""))
            if msg.get("role")=="assistant" and msg.get("audio"): play_hidden_audio(msg.get("audio"))
            if msg.get("sources"):
                with st.expander("📚 RAG Source"):
                    for s in msg["sources"]: st.caption(s[:600])
prompt = st.chat_input("Message MANHATTAN... ya Mofisto lai bolau!")
if "pending_prompt" in st.session_state: prompt = st.session_state.pop("pending_prompt")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="🧑"): st.markdown(prompt)
    want_image = (mode=="🎨 Image Generator") or is_image_request(prompt)
    want_video = (mode=="🎬 Video Generator") or is_video_request(prompt)
    if st.session_state.bot_enabled:
        st.session_state.bot_message = random.choice(BOT_PERSONALITIES[st.session_state.bot_name]['tips'])
    if want_video:
        clean_prompt=prompt.replace("video bana","").replace("video banau","").replace("vedio bana","").replace("🎬","").replace("🎥","").strip()
        if mode=="🎬 Video Generator":
            style_map_v={"Realistic":"realistic, ultra detailed","Anime":"anime style, vibrant","3D Cartoon":"3D pixar style cartoon","Cinematic":"cinematic, dramatic lighting","Cyberpunk":"cyberpunk, neon lights","Slow Motion":"slow motion, dreamy"}
            clean_prompt=f"{clean_prompt}, {style_map_v.get(vid_style,'')}"; dur=vid_duration; asp=vid_aspect
        else: dur=5; asp="16:9"
        if not clean_prompt: clean_prompt=prompt
        vid_url=make_video_url(clean_prompt, duration=dur, aspect=asp)
        with st.chat_message("assistant", avatar="🔥"):
            with st.spinner("🎬 Video banaudai cha..."):
                st.markdown(f"La timro video banayo! 🔥\n**Prompt:** `{clean_prompt}`")
                st.video(vid_url); st.link_button("⬇️ Video Download", vid_url, use_container_width=True)
                audio_bytes = text_to_speech("Here is your video, check it out!") if st.session_state.voice_reply else None
                if audio_bytes: play_hidden_audio(audio_bytes)
                st.session_state.messages.append({"role":"assistant","content":f"La timro video banayo! 🔥\n**Prompt:** `{clean_prompt}`","type":"video","video_url":vid_url,"audio":audio_bytes})
                st.session_state.current_chat_id = save_current_chat(st.session_state.username, [{k:v for k,v in m.items() if k!="audio"} for m in st.session_state.messages], st.session_state.current_chat_id)
    elif want_image:
        clean_prompt=prompt.replace("photo bana","").replace("photo banau","").replace("image bana","").replace("🎨","").strip()
        if mode=="🎨 Image Generator":
            style_map={"Realistic Photo":"realistic photo, ultra detailed","Anime":"anime style, vibrant","3D Cartoon":"3D pixar style cartoon","Cyberpunk":"cyberpunk, neon lights","Oil Painting":"oil painting, artistic","Cinematic":"cinematic, dramatic lighting"}
            clean_prompt=f"{clean_prompt}, {style_map.get(img_style,'')}"
        if not clean_prompt: clean_prompt=prompt
        img_url=make_image_url(clean_prompt)
        with st.chat_message("assistant", avatar="🔥"):
            with st.spinner("🎨 Photo banaudai cha..."):
                st.markdown(f"La timro photo banayo! 🔥\n**Prompt:** `{clean_prompt}`")
                st.image(img_url, use_container_width=True); st.link_button("⬇️ Photo Download", img_url, use_container_width=True)
                audio_bytes = text_to_speech("Here is your photo, look how nice it is!") if st.session_state.voice_reply else None
                if audio_bytes: play_hidden_audio(audio_bytes)
                st.session_state.messages.append({"role":"assistant","content":f"La timro photo banayo! 🔥\n**Prompt:** `{clean_prompt}`","type":"image","image_url":img_url,"audio":audio_bytes})
                st.session_state.current_chat_id = save_current_chat(st.session_state.username, [{k:v for k,v in m.items() if k!="audio"} for m in st.session_state.messages], st.session_state.current_chat_id)
    else:
        if not api_key: st.error("Secrets ma API Key chaina!")
        else:
            rag_context=""; sources=[]
            if st.session_state.rag_chunks:
                relevant=retrieve_relevant(prompt, st.session_state.rag_chunks, top_k=3)
                sources=relevant; rag_context="\n\n".join(relevant)
                SYSTEM_PROMPT=BASE_PROMPT+f"\n\n--- RAG CONTEXT ---\n{rag_context}\n--- END CONTEXT ---\n\nNiyam: Yei CONTEXT bata answer deu."
            else: SYSTEM_PROMPT=BASE_PROMPT
            all_msgs=[SystemMessage(content=SYSTEM_PROMPT)]
            for msg in st.session_state.messages:
                if msg.get("type") in ["image","video"]: continue
                if msg["role"]=="user": all_msgs.append(HumanMessage(content=msg["content"]))
                elif msg["role"]=="assistant": all_msgs.append(AIMessage(content=msg["content"]))
            with st.chat_message("assistant", avatar="🔥"):
                with st.spinner("MANHATTAN is thinking..."):
                    try:
                        llm=ChatOpenAI(model=model_choice.strip(), openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1", temperature=temperature)
                        res=llm.invoke(all_msgs); answer=res.content
                        if isinstance(answer, list): answer="".join(p.get("text","") if isinstance(p, dict) else str(p) for p in answer)
                        answer=str(answer); st.markdown(answer)
                        audio_bytes=None
                        if st.session_state.voice_reply:
                            with st.spinner("🔊 Voice banaudai..."):
                                audio_bytes = text_to_speech(answer)
                                if audio_bytes: play_hidden_audio(audio_bytes)
                        if sources:
                            with st.expander("📚 RAG Source"):
                                for s in sources: st.caption(s[:600])
                        save_msg = {"role":"assistant","content":answer,"sources":sources if sources else None}
                        st.session_state.messages.append(save_msg)
                        st.session_state.messages[-1]["audio"] = audio_bytes
                        st.session_state.current_chat_id = save_current_chat(st.session_state.username, [{k:v for k,v in m.items() if k!="audio"} for m in st.session_state.messages], st.session_state.current_chat_id)
                    except Exception as e:
                        st.error("Couldn't connect to OpenRouter."); st.caption(f"Details: {e}")
st.markdown('<div class="footer">MANHATTAN AI · RAG + Chat + Image + Video + Login + Voice + Mofisto · Made in Nepal 🇳🇵</div>', unsafe_allow_html=True)
