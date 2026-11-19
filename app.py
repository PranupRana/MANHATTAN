import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import datetime, urllib.parse, random
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

st.set_page_config(page_title="MANHATTAN 🔥", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

# ---------- PREMIUM CSS ----------
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stApp { background: linear-gradient(135deg, #0E1117 0%, #1A1B2E 100%); }
    .prompt-card { background: #1E2130; padding: 16px; border-radius: 16px; border: 1px solid #2E3440; text-align: center; margin-bottom: 10px; transition: 0.3s; }
    .prompt-card:hover { border-color: #FF4B4B; transform: translateY(-2px); }
    .prompt-title { font-size: 16px; font-weight: 700; color: white; }
    .prompt-desc { font-size: 12px; color: #9AA0B0; margin-top: 4px; }
    .footer { text-align: center; color: #666; font-size: 12px; margin-top: 40px; }
    .manhattan-logo { font-size: 32px; font-weight: 900; background: linear-gradient(90deg, #FF4B4B, #FF8A00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stChatMessage"] { border-radius: 16px; background-color: #232838 !important; border: 1px solid #3A4157 !important; }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li, div[data-testid="stChatMessage"] span {
        color: #FFFFFF !important; font-size: 16px !important; line-height: 1.6 !important; opacity: 1 !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label { color: #E0E0E0 !important; }
    h1, h2, h3 { color: white !important; }
    .rag-box { background: #1A2E1A; padding: 12px; border-radius: 12px; border: 1px solid #2E7D32; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# ---------- RAG FUNCTIONS ----------
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
    return [c.strip() for c in chunks if len(c.strip()) > 50]

def extract_text(file):
    if file.name.endswith(".pdf"):
        if not HAS_FITZ: return ""
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    elif file.name.endswith(".txt") or file.name.endswith(".md"):
        return file.read().decode("utf-8", errors="ignore")
    return ""

def retrieve_relevant(query, chunks, top_k=3):
    if not chunks or not HAS_SKLEARN:
        return chunks[:top_k]
    try:
        vectorizer = TfidfVectorizer().fit(chunks + [query])
        vecs = vectorizer.transform(chunks + [query])
        sims = cosine_similarity(vecs[-1], vecs[:-1])[0]
        idx = np.argsort(sims)[::-1][:top_k]
        return [chunks[i] for i in idx if sims[i] > 0.05]
    except:
        return chunks[:top_k]

# ---------- HELPER - PHOTO BANAUNE ----------
def is_image_request(text):
    text = text.lower()
    keywords = ["photo", "image", "picture", "pic ", "draw", "bana", "banau", "banai", "tasbir", "chitra", "poster", "wallpaper", "logo", "generate image", "/image", "🎨"]
    return any(k in text for k in keywords)

def make_image_url(prompt):
    clean = urllib.parse.quote(prompt)
    seed = random.randint(1, 999999)
    return f"https://image.pollinations.ai/prompt/{clean}?width=1024&height=1024&seed={seed}&nologo=true&model=flux"

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []
if "rag_files" not in st.session_state:
    st.session_state.rag_files = []

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown('<div class="manhattan-logo">MANHATTAN 🔥</div>', unsafe_allow_html=True)
    st.caption("Don't use others AI. I built my own.")
    st.divider()

    # --- RAG SECTION - NAYA ---
    st.subheader("📚 RAG - Apno Document")
    st.caption("PDF / TXT halnu, MANHATTAN le tyai bata answer dincha!")
    uploaded = st.file_uploader("📄 Document halnu (PDF / TXT)", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.rag_files:
                text = extract_text(f)
                if text:
                    chunks = chunk_text(text)
                    st.session_state.rag_chunks.extend(chunks)
                    st.session_state.rag_files.append(f.name)
                    st.success(f"✅ {f.name} - {len(chunks)} chunks bho!")
    if st.session_state.rag_chunks:
        st.markdown(f'<div class="rag-box">📚 <b>{len(st.session_state.rag_files)} file</b> loaded<br>🧩 {len(st.session_state.rag_chunks)} chunks ready<br>✅ RAG ON cha!</div>', unsafe_allow_html=True)
        if st.button("🗑️ RAG Documents Clear Garne", use_container_width=True):
            st.session_state.rag_chunks = []
            st.session_state.rag_files = []
            st.rerun()
    else:
        st.info("💡 RAG test garna: Kunai PDF halnu, ani tyo PDF bare sodhnu!")

    st.divider()
    mode = st.radio("✨ Mode", ["💬 Chat", "🎨 Image Generator"], index=0, horizontal=True)
    st.divider()
    st.subheader("⚙️ Settings")
    model_choice = st.selectbox("🧠 Model", [
        "openrouter/free",
        "nvidia/nemotron-3.5-super-vl:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "cohere/north-mini-code:free",
        "google/gemma-4-31b-it:free",
    ], index=0)
    temperature = st.slider("🔥 Creativity (Temperature)", 0.0, 1.5, 0.7, 0.1)
    personality = st.selectbox("🎭 Personality", ["Bold Friend", "Teacher", "Coder Bro", "Roaster 😈"])
    if mode == "🎨 Image Generator":
        st.subheader("🎨 Image Style")
        img_style = st.selectbox("Style", ["Realistic Photo", "Anime", "3D Cartoon", "Cyberpunk", "Oil Painting", "Cinematic"])
    st.divider()
    st.subheader("📊 Chat Stats")
    st.metric("Total Messages", len(st.session_state.messages))
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    if st.session_state.messages:
        chat_text = "\n\n".join([f"{m['role'].upper()}: {m.get('content','')}" for m in st.session_state.messages])
        st.download_button("⬇️ Download Chat", chat_text, file_name=f"manhattan-{datetime.date.today()}.txt", use_container_width=True)
    st.divider()
    st.caption("Made with ❤️ in Nepal by PranupRana")
    st.caption("Powered by OpenRouter + Pollinations + RAG")

PERSONALITY_PROMPTS = {
    "Bold Friend": "You are MANHATTAN. You answer any question directly and honestly. No lecture, no moralizing. Speak like a real bold friend from Nepal, mix Nepali + English.",
    "Teacher": "You are MANHATTAN, a smart teacher. Explain everything simply, with examples, in easy Nepali-English mix.",
    "Coder Bro": "You are MANHATTAN, an expert coder. Give clean code, explain in short, no extra bakwas.",
    "Roaster 😈": "You are MANHATTAN, funny and savage. Roast lightly, be hilarious, but still answer correctly.",
}
BASE_PROMPT = PERSONALITY_PROMPTS[personality]

# ---------- MAIN ----------
st.title("MANHATTAN 🔥")
if st.session_state.rag_chunks:
    st.success(f"📚 RAG ON: {', '.join(st.session_state.rag_files)} bata answer dinchu!")
if mode == "🎨 Image Generator":
    st.caption("🎨 **Image Mode ON** - Je lekchau, tyo photo bancha! Eg: `A boy in Kathmandu at sunset`")
else:
    st.caption(f"Chatting with **{personality}** • Model: `{model_choice}` • RAG: {'ON 🟢' if st.session_state.rag_chunks else 'OFF - PDF halnu'} • Photo chahiyo bhane `photo bana` lekha!")

if len(st.session_state.messages) == 0:
    st.markdown("### 👋 Ke garne? Choose gara:")
    prompts = [
        ("📚 RAG Test", "PDF bata sodha", "Yo document ma ke cha? Summary deu."),
        ("🎨 Photo Banau", "A boy in Himalayas, cinematic", "🎨 A boy standing in Himalayas at sunset, cinematic, realistic photo"),
        ("💻 Coding Help", "Debug code or learn programming", "Help me with a coding problem."),
        ("💡 Business Idea", "Give me a business idea for Nepal", "Give me 5 business ideas I can start in Nepal with low money."),
    ]
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]
    for i, (title, desc, prompt_text) in enumerate(prompts):
        with cols[i]:
            st.markdown(f'<div class="prompt-card"><div class="prompt-title">{title}</div><div class="prompt-desc">{desc}</div></div>', unsafe_allow_html=True)
            if st.button("Use →", key=f"suggest{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

# Show History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🔥"):
        if msg.get("type") == "image":
            st.markdown(msg.get("content", ""))
            st.image(msg.get("image_url"), use_container_width=True)
        else:
            st.markdown(msg.get("content", ""))
            if msg.get("sources"):
                with st.expander("📚 RAG Source - Kaha bata lyaeko?"):
                    for s in msg["sources"]:
                        st.caption(s[:600])

prompt = st.chat_input("Message MANHATTAN... RAG cha bhane PDF bare sodha!")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    want_image = (mode == "🎨 Image Generator") or is_image_request(prompt)

    if want_image:
        clean_prompt = prompt.replace("photo bana", "").replace("photo banau", "").replace("image bana", "").replace("🎨", "").strip()
        if mode == "🎨 Image Generator":
            style_map = {"Realistic Photo": "realistic photo, ultra detailed", "Anime": "anime style, vibrant", "3D Cartoon": "3D pixar style cartoon", "Cyberpunk": "cyberpunk, neon lights", "Oil Painting": "oil painting, artistic", "Cinematic": "cinematic, dramatic lighting"}
            clean_prompt = f"{clean_prompt}, {style_map.get(img_style, '')}"
        if not clean_prompt:
            clean_prompt = prompt
        img_url = make_image_url(clean_prompt)
        with st.chat_message("assistant", avatar="🔥"):
            with st.spinner("🎨 MANHATTAN le photo banaudai cha... 10 sec parkha!"):
                st.markdown(f"La timro photo banayo! 🔥\n**Prompt:** `{clean_prompt}`")
                st.image(img_url, use_container_width=True)
                st.link_button("⬇️ Photo Download Garne", img_url, use_container_width=True)
                st.session_state.messages.append({"role": "assistant", "content": f"La timro photo banayo! 🔥\n**Prompt:** `{clean_prompt}`", "type": "image", "image_url": img_url})
    else:
        if not api_key:
            st.error("Secrets ma API Key chaina!")
        else:
            # --- RAG LOGIC YAHAN ---
            rag_context = ""
            sources = []
            if st.session_state.rag_chunks:
                relevant = retrieve_relevant(prompt, st.session_state.rag_chunks, top_k=3)
                sources = relevant
                rag_context = "\n\n".join(relevant)
                SYSTEM_PROMPT = BASE_PROMPT + f"\n\n--- RAG CONTEXT (User ko Document bata) ---\n{rag_context}\n--- END CONTEXT ---\n\nNiyam: User le sodheko kura yei CONTEXT bata answer deu. Context ma navaye 'Yo kura timro document ma chaina' bhanera bhanne, afno man le invent nagarne."
            else:
                SYSTEM_PROMPT = BASE_PROMPT

            all_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
            for msg in st.session_state.messages:
                if msg.get("type") == "image":
                    continue
                if msg["role"] == "user":
                    all_msgs.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    all_msgs.append(AIMessage(content=msg["content"]))
            with st.chat_message("assistant", avatar="🔥"):
                with st.spinner("MANHATTAN is thinking... RAG search gardai..."):
                    try:
                        llm = ChatOpenAI(model=model_choice.strip(), openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1", temperature=temperature)
                        res = llm.invoke(all_msgs)
                        answer = res.content
                        if isinstance(answer, list):
                            answer = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in answer)
                        answer = str(answer)
                        st.markdown(answer)
                        if sources:
                            with st.expander("📚 RAG Source - Kaha bata lyaeko?"):
                                for s in sources:
                                    st.caption(s[:600])
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources if sources else None})
                    except Exception as e:
                        st.error("Couldn't connect to OpenRouter. Check API key / model / rate limit.")
                        st.caption(f"Details: {e}")

st.markdown('<div class="footer">MANHATTAN AI · RAG + Chat + Image · Made in Nepal 🇳🇵</div>', unsafe_allow_html=True)
