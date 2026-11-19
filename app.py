import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import datetime

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
    div[data-testid="stChatMessage"] { border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR - SMART ----------
with st.sidebar:
    st.markdown('<div class="manhattan-logo">MANHATTAN 🔥</div>', unsafe_allow_html=True)
    st.caption("Don't use others AI. I built my own.")
    st.divider()

    # --- API KEY ---
    st.subheader("🔑 API Key")
    _secret_key = ""
    try:
        _secret_key = st.secrets.get("OPENROUTER_API_KEY", "")
    except:
        pass
    api_key = st.text_input("OpenRouter API Key", value=_secret_key, type="password", help="openrouter.ai > Keys bata lyaera yaha paste gara")
    if not api_key:
        st.warning("⚠️ API Key hala natra chat chaldaina!")
    st.divider()

    st.subheader("⚙️ Settings")
    # --- CHALNE MODEL HARU MATRA ---
    model_choice = st.selectbox("🧠 Model", [
        "openrouter/free",
        "nvidia/nemotron-3.5-super-vl:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "cohere/north-mini-code:free",
        "google/gemma-4-31b-it:free",
    ], index=0)
    
    temperature = st.slider("🔥 Creativity (Temperature)", 0.0, 1.5, 0.7, 0.1, help="Low = Serious, High = Pagal / Funny")
    personality = st.selectbox("🎭 Personality", ["Bold Friend", "Teacher", "Coder Bro", "Roaster 😈"])
    
    st.divider()
    st.subheader("📊 Chat Stats")
    if "messages" not in st.session_state:
        st.session_state.messages = []
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
        chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("⬇️ Download Chat", chat_text, file_name=f"manhattan-{datetime.date.today()}.txt", use_container_width=True)

    st.divider()
    st.caption("Made with ❤️ in Nepal by PranupRana")
    st.caption("Powered by OpenRouter")

# ---------- SYSTEM PROMPT ----------
PERSONALITY_PROMPTS = {
    "Bold Friend": "You are MANHATTAN. You answer any question directly and honestly. No lecture, no moralizing. Speak like a real bold friend from Nepal, mix Nepali + English.",
    "Teacher": "You are MANHATTAN, a smart teacher. Explain everything simply, with examples, in easy Nepali-English mix.",
    "Coder Bro": "You are MANHATTAN, an expert coder. Give clean code, explain in short, no extra bakwas.",
    "Roaster 😈": "You are MANHATTAN, funny and savage. Roast lightly, be hilarious, but still answer correctly.",
}
SYSTEM_PROMPT = PERSONALITY_PROMPTS[personality]

# ---------- MAIN AREA ----------
st.title("MANHATTAN 🔥")
st.caption(f"Chatting with **{personality}** • Model: `{model_choice}`")

if len(st.session_state.messages) == 0:
    st.markdown("### 👋 Ke garne? Choose gara:")
    prompts = [
        ("📚 Learn Something", "Explain a difficult concept simply", "Explain a difficult concept simply."),
        ("✍️ Writing Assistant", "Write, improve, or rewrite text", "Help me write or improve some text."),
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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🔥"):
        st.markdown(msg["content"])

prompt = st.chat_input("Message MANHATTAN... (Shift+Enter for new line)")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    if not api_key:
        st.error("Pahila Sidebar ma API Key hala!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        all_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                all_msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                all_msgs.append(AIMessage(content=msg["content"]))

        with st.chat_message("assistant", avatar="🔥"):
            with st.spinner("MANHATTAN is thinking..."):
                try:
                    llm = ChatOpenAI(
                        model=model_choice.strip(),
                        openai_api_key=api_key,
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=temperature,
                    )
                    res = llm.invoke(all_msgs)
                    answer = res.content
                    if isinstance(answer, list):
                        answer = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in answer)
                    answer = str(answer)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error("Couldn't connect to OpenRouter. Check API key / model / rate limit.")
                    st.caption(f"Details: {e}")

st.markdown('<div class="footer">MANHATTAN AI · Powered by OpenRouter · Made in Nepal 🇳🇵</div>', unsafe_allow_html=True)
