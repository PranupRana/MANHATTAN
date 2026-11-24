import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import datetime, urllib.parse, random, hashlib, sqlite3, json, io
import numpy as np

# --- RAG ko lagi ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False
try:
    import fitz
    HAS_FITZ = True
except:
    HAS_FITZ = False

# --- VOICE ko lagi ---
try:
    from gtts import gTTS
    HAS_TTS = True
except:
    HAS_TTS = False
try:
    import speech_recognition as sr
    HAS_STT = True
except:
    HAS_STT = False

st.set_page_config(page_title="MANHATTAN 🔥", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stApp { background: linear-gradient(135deg, #0E1117 0%, #1A1B2E 100%); }
    .prompt-card { background: #1E2130; padding: 16px; border-radius: 16px; border: 1px solid #2E3440; text-align: center; margin-bottom: 10px; }
    .prompt-title { font-size: 16px; font-weight: 700; color: white; }
    .prompt-desc { font-size: 12px; color: #9AA0B0; margin-top: 4px; }
    .footer { text-align: center; color: #666; font-size: 12px; margin-top: 40px; }
    .manhattan-logo { font-size: 32px; font-weight: 900; background: linear-gradient(90deg, #FF4B4B, #FF8A00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stChatMessage"] { border-radius: 16px; background-color: #232838 !important; border: 1px solid #3A4157 !important; }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li, div[data-testid="stChatMessage"] span { color: #FFFFFF !important; font-size: 16px !important; line-height: 1.6 !important; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label { color: #E0E0E0 !important; }
    h1, h2, h3 { color: white !important; }
    .rag-box { background: #1A2E1A; padding: 12px; border-radius: 12px; border: 1px solid #2E7D32; margin-bottom: 10px; }
    .voice-box { background: #2A1A3E; padding: 12px; border-radius: 12px; border: 1px solid #7B2FFF; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# ==================== 🎤 VOICE FUNCTIONS ====================
def text_to_speech(text, lang="en"):
    """Text -> Audio (gTTS) - returns bytes"""
    if not HAS_TTS:
        return None
    # Markdown / emoji hataune, 800 char samma matra bolne
    import re
    clean = re.sub(r'[#*`_~>\[\]()|]', '', text)
    clean = re.sub(r'[^\w\s.,!?\'-]', ' ', clean)
    clean = clean[:800]
    if not clean.strip():
        return None
    try:
        # Nepali mix bhaye english ma bolne, pure nepali detect gare nepali
        if any(ord(c) > 127 and '\u0900' <= c <= '\u097F' for c in text):
            lang = "ne"
        tts = gTTS(text=clean, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print("TTS error:", e)
        return None

def speech_to_text(audio_bytes):
    """Audio -> Text (Google Speech Recognition)"""
    if not HAS_STT:
        return None, "SpeechRecognition install chaina!"
    try:
        r = sr.Recognizer()
        # audio_bytes lai wav ma convert
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = r.record(source)
            # English + Nepali try
            try:
                text = r.recognize_google(audio, language="en-IN")
            except:
                text = r.recognize_google(audio, language="ne-NP")
            return text, None
    except sr.UnknownValueError:
        return None, "Awaz bujhina, feri bolna! 🎤"
    except Exception as e:
        return None, f"Voice error: {e}"

# ==================== 🔐 LOGIN + HISTORY DB ====================
DB_PATH = "manhattan.db"
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT, created_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, title TEXT, messages TEXT, updated_at TEXT)""")
    conn.commit()
    return conn
def hash_pass(pw): return hashlib.sha256(pw.encode()).hexdigest()
def signup_user(username, password):
    conn = get_db()
    try:
        conn.execute("INSERT INTO users VALUES (?,?,?)", (username, hash_pass(password), str(datetime.datetime.now())))
        conn.commit(); return True, "Account banayo! Abba Login gara."
    except sqlite3.IntegrityError:
        return False, "Yo username pahila nai cha!"
    finally: conn.close()
def login_user(username, password):
    conn = get_db()
    row = conn.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return row and row[0] == hash_pass(password)
def save_current_chat(username, messages, chat_id=None, title=None):
    if not messages: return chat_id
    if not title:
        first = next((m.get("content","") for m in messages if m.get("role")=="user"), "New Chat")
        title = first[:40] + ("..." if len(first) > 40 else "")
    conn = get_db(); now = str(datetime.datetime.now())
    if chat_id:
        conn.execute("UPDATE chats SET title=?, messages=?, updated_at=? WHERE id=? AND username=?", (title, json.dumps(messages), now, chat_id, username))
    else:
        cur = conn.execute("INSERT INTO chats (username,title,messages,updated_at) VALUES (?,?,?,?)", (username, title, json.dumps(messages), now))
        chat_id = cur.lastrowid
    conn.commit(); conn.close(); return chat_id
def get_user_chats(username):
    conn = get_db()
    rows = conn.execute("SELECT id, title, updated_at FROM chats WHERE username=? ORDER BY updated_at DESC", (username,)).fetchall()
    conn.close(); return rows
def load_chat(chat_id, username):
    conn = get_db()
    row = conn.execute("SELECT messages FROM chats WHERE id=? AND username=?", (chat_id, username)).fetchone()
    conn.close()
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
if "voice_lang" not in st.session_state: st.session_state.voice_lang = "Auto"

# ==================== 🔐 LOGIN PAGE ====================
if not st.session_state.logged_in:
    st.markdown('<div class="manhattan-logo" style="text-align:center;">MANHATTAN 🔥</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#9AA0B0;'>Don't use others AI. I built my own.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔐 Login", "✨ Signup"])
        with tab_login:
            st.subheader("Welcome Back 🔥")
            lu = st.text_input("Username", key="login_u")
            lp = st.text_input("Password", type="password", key="login_p")
            if st.button("Login", use_container_width=True, type="primary"):
                if not lu or not lp: st.error("Username + Password duitai halna!")
                elif login_user(lu.strip(), lp):
                    st.session_state.logged_in = True; st.session_state.username = lu.strip()
                    st.session_state.messages = []; st.session_state.current_chat_id = None
                    st.success(f"Welcome back, {lu}! 🔥"); st.rerun()
                else: st.error("Username / Password galat cha!")
        with tab_signup:
            st.subheader("Naya Account Banau")
            su = st.text_input("Naya Username", key="signup_u")
            sp = st.text_input("Naya Password", type="password", key="signup_p")
            sp2 = st.text_input("Password Confirm", type="password", key="signup_p2")
            if st.button("Signup Garne", use_container_width=True, type="primary"):
                if not su or not sp: st.error("Username + Password chahincha!")
                elif sp != sp2: st.error("Password milena!")
                elif len(sp) < 4: st.error("Password kamti ma 4 character hunu parcha!")
                else:
                    ok, msg = signup_user(su.strip(), sp)
                    st.success(msg) if ok else st.error(msg)
    st.markdown('<div class="footer">MANHATTAN AI · Made in Nepal 🇳🇵</div>', unsafe_allow_html=True)
    st.stop()

# ==================== MAIN APP ====================
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
    elif file.name.endswith(".txt") or file.name.endswith(".md"):
        return file.read().decode("utf-8", errors="ignore")
    return ""
def retrieve_relevant(query, chunks, top_k=3):
    if not chunks or not HAS_SKLEARN: return chunks[:top_k]
    try:
        vectorizer=TfidfVectorizer().fit(chunks+[query]); vecs=vectorizer.transform(chunks+[query])
        sims=cosine_similarity(vecs[-1], vecs[:-1])[0]; idx=np.argsort(sims)[::-1][:top_k]
        return [chunks[i] for i in idx if sims[i]>0.05]
    except: return chunks[:top_k]
def is_image_request(text):
    text=text.lower(); keywords=["photo","image","picture","pic ","draw","bana","banau","banai","tasbir","chitra","poster","wallpaper","logo","generate image","/image","🎨"]
    return any(k in text for k in keywords)
def is_video_request(text):
    text=text.lower(); keywords=["video","vedio","bhidiyo","reel","clip","animation","animate","motion","/video","🎬","🎥"]
    return any(k in text for k in keywords)
def make_image_url(prompt):
    clean=urllib.parse.quote(prompt); seed=random.randint(1,999999)
    return f"https://image.pollinations.ai/prompt/{clean}?width=1024&height=1024&seed={seed}&nologo=true&model=flux"
def make_video_url(prompt, duration=5, aspect="16:9"):
    clean=urllib.parse.quote(prompt); seed=random.randint(1,999999)
    w,h=(1280,720) if aspect=="16:9" else (720,1280) if aspect=="9:16" else (1024,1024)
    return f"https://gen.pollinations.ai/video/{clean}?width={w}&height={h}&seed={seed}&duration={duration}&nologo=true&model=veo"

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown('<div class="manhattan-logo">MANHATTAN 🔥</div>', unsafe_allow_html=True)
    st.caption(f"👤 Logged in: **{st.session_state.username}**")
    if st.button("🚪 Logout", use_container_width=True):
        if st.session_state.messages:
            st.session_state.current_chat_id = save_current_chat(st.session_state.username, st.session_state.messages, st.session_state.current_chat_id)
        st.session_state.logged_in=False; st.session_state.username=""; st.session_state.messages=[]; st.session_state.current_chat_id=None; st.rerun()
    st.divider()

    # --- 🎤 VOICE SETTINGS - NAYA ---
    st.subheader("🎤 Voice Settings")
    st.markdown('<div class="voice-box">🎙️ Bolera sodhna, sunera bujhna milcha!</div>', unsafe_allow_html=True)
    voice_enabled = st.toggle("🔊 Voice Reply ON (MANHATTAN le bolcha)", value=st.session_state.voice_reply)
    st.session_state.voice_reply = voice_enabled
    voice_lang_opt = st.selectbox("🗣️ Voice Language", ["Auto", "English", "Nepali"], index=0)
    voice_speed = st.selectbox("⏩ Voice Speed", ["Normal", "Slow"], index=0)
    st.divider()

    # --- 🎤 VOICE INPUT ---
    st.subheader("🎤 Voice Input - Bolera Sodha")
    audio_input = st.audio_input("🎙️ Yaha thichera bola...", key="voice_input_main")
    if audio_input is not None:
        if st.button("🎤 Yo Awaz lai Text Banau", use_container_width=True, type="primary"):
            with st.spinner("Awaz bujhdai chu..."):
                audio_bytes = audio_input.read()
                text, err = speech_to_text(audio_bytes)
                if text:
                    st.success(f"Timle bhaneko: **{text}**")
                    st.session_state.pending_prompt = text
                    st.rerun()
                else:
                    st.error(err or "Awaz bujhina!")
    st.divider()

    st.subheader("📂 Chat History")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        if st.session_state.messages:
            st.session_state.current_chat_id = save_current_chat(st.session_state.username, st.session_state.messages, st.session_state.current_chat_id)
        st.session_state.messages=[]; st.session_state.current_chat_id=None; st.rerun()
    user_chats = get_user_chats(st.session_state.username)
    if user_chats:
        for cid, title, updated in user_chats:
            c1, c2 = st.columns([4,1])
            with c1:
                if st.button(f"💬 {title}", key=f"hist_{cid}", use_container_width=True):
                    if st.session_state.messages:
                        st.session_state.current_chat_id = save_current_chat(st.session_state.username, st.session_state.messages, st.session_state.current_chat_id)
                    st.session_state.messages = load_chat(cid, st.session_state.username); st.session_state.current_chat_id=cid; st.rerun()
            with c2:
                if st.button("🗑️", key=f"del_{cid}"):
                    delete_chat(cid, st.session_state.username)
                    if st.session_state.current_chat_id==cid: st.session_state.messages=[]; st.session_state.current_chat_id=None
                    st.rerun()
    else:
        st.info("Aile samma chat chaina.")
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
    if mode=="🎨 Image Generator":
        img_style = st.selectbox("Style", ["Realistic Photo","Anime","3D Cartoon","Cyberpunk","Oil Painting","Cinematic"])
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

# ---------- MAIN ----------
st.title("MANHATTAN 🔥")
st.caption(f"Namaste, **{st.session_state.username}** 👋 • 🎤 Bolera sodhna sidebar ma Mic use gara!")
if st.session_state.rag_chunks:
    st.success(f"📚 RAG ON: {', '.join(st.session_state.rag_files)} bata answer dinchu!")
if mode=="🎨 Image Generator": st.caption("🎨 **Image Mode ON**")
elif mode=="🎬 Video Generator": st.caption("🎬 **Video Mode ON** - 30-60 sec lagcha!")
else: st.caption(f"Chatting with **{personality}** • Model: `{model_choice}`")

if len(st.session_state.messages)==0:
    st.markdown("### 👋 Ke garne? Choose gara:")
    prompts=[("📚 RAG Test","PDF bata sodha","Yo document ma ke cha? Summary deu."),("🎨 Photo Banau","A boy in Himalayas","🎨 A boy standing in Himalayas at sunset, cinematic"),("🎬 Video Banau","A boy running in rain","🎬 A boy running in Kathmandu rain, cinematic"),("🎤 Voice Test","Bolera sodha","Hello MANHATTAN, timi ko hau?")]
    c1,c2,c3,c4=st.columns(4); cols=[c1,c2,c3,c4]
    for i,(title,desc,prompt_text) in enumerate(prompts):
        with cols[i]:
            st.markdown(f'<div class="prompt-card"><div class="prompt-title">{title}</div><div class="prompt-desc">{desc}</div></div>', unsafe_allow_html=True)
            if st.button("Use →", key=f"suggest{i}", use_container_width=True):
                st.session_state.pending_prompt=prompt_text; st.rerun()

# Show History - Voice pani dekhaune
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"]=="user" else "🔥"):
        if msg.get("type")=="image":
            st.markdown(msg.get("content","")); st.image(msg.get("image_url"), use_container_width=True)
        elif msg.get("type")=="video":
            st.markdown(msg.get("content","")); st.video(msg.get("video_url")); st.link_button("⬇️ Video Download", msg.get("video_url"), use_container_width=True)
        else:
            st.markdown(msg.get("content",""))
            # Purano chat ma pani voice sunna milos - Voice Reply ON cha bhane
            if msg.get("role")=="assistant" and msg.get("audio"):
                st.audio(msg.get("audio"), format="audio/mp3")
            if msg.get("sources"):
                with st.expander("📚 RAG Source"):
                    for s in msg["sources"]: st.caption(s[:600])

prompt = st.chat_input("Message MANHATTAN... ya Sidebar ma 🎤 bolera sodha!")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="🧑"): st.markdown(prompt)
    want_image = (mode=="🎨 Image Generator") or is_image_request(prompt)
    want_video = (mode=="🎬 Video Generator") or is_video_request(prompt)

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
                # Video ko lagi pani voice
                voice_txt = "La timro video banayo, hera ta!"
                audio_bytes = text_to_speech(voice_txt) if st.session_state.voice_reply else None
                if audio_bytes: st.audio(audio_bytes, format="audio/mp3")
                st.session_state.messages.append({"role":"assistant","content":f"La timro video banayo! 🔥\n**Prompt:** `{clean_prompt}`","type":"video","video_url":vid_url,"audio":audio_bytes})
                st.session_state.current_chat_id = save_current_chat(st.session_state.username, st.session_state.messages, st.session_state.current_chat_id)
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
                voice_txt = "La timro photo banayo, hera ta katti ramro!"
                audio_bytes = text_to_speech(voice_txt) if st.session_state.voice_reply else None
                if audio_bytes: st.audio(audio_bytes, format="audio/mp3")
                st.session_state.messages.append({"role":"assistant","content":f"La timro photo banayo! 🔥\n**Prompt:** `{clean_prompt}`","type":"image","image_url":img_url,"audio":audio_bytes})
                st.session_state.current_chat_id = save_current_chat(st.session_state.username, st.session_state.messages, st.session_state.current_chat_id)
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
                        answer=str(answer)
                        st.markdown(answer)
                        # 🎤 VOICE REPLY - NAYA
                        audio_bytes=None
                        if st.session_state.voice_reply:
                            with st.spinner("🔊 Voice banaudai..."):
                                lang_code = "ne" if voice_lang_opt=="Nepali" else "en" if voice_lang_opt=="English" else "en"
                                audio_bytes = text_to_speech(answer, lang=lang_code)
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                        if sources:
                            with st.expander("📚 RAG Source"):
                                for s in sources: st.caption(s[:600])
                        # Audio lai DB ma save garna sakdaina (bytes json ma mildaina) - isliye audio lai session ma matra rakhne, DB ma audio=None
                        save_msg = {"role":"assistant","content":answer,"sources":sources if sources else None}
                        st.session_state.messages.append(save_msg)
                        # Display ko lagi audio pani session ma alag bata rakhne
                        st.session_state.messages[-1]["audio"] = audio_bytes
                        st.session_state.current_chat_id = save_current_chat(st.session_state.username, [{k:v for k,v in m.items() if k!="audio"} for m in st.session_state.messages], st.session_state.current_chat_id)
                    except Exception as e:
                        st.error("Couldn't connect to OpenRouter."); st.caption(f"Details: {e}")

st.markdown('<div class="footer">MANHATTAN AI · RAG + Chat + Image + Video + Login + Voice · Made in Nepal 🇳🇵</div>', unsafe_allow_html=True)
