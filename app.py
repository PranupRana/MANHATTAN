import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="MANHATTAN - AI ChatBot", page_icon="🤖")
st.title("🤖 MANHATTAN")
st.caption("Streamlit + LangChain + OpenRouter | Built by PranupRana")

# Sidebar - Key halne thau
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
st.sidebar.markdown("---")
st.sidebar.write("**Project:** MANHATTAN")
st.sidebar.write("**Tech:** Streamlit + LangChain")

if not api_key:
    st.info("👈 Pahila sidebar ma afno OpenRouter API Key hala, ani chat suru huncha!")
    st.stop()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaste! Ma MANHATTAN hu. Kehi sodha! 🙌"}
    ]

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if prompt := st.chat_input("Kehi sodha..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Sochdai chu..."):
            res = llm.invoke([HumanMessage(content=prompt)])
            st.write(res.content)
            st.session_state.messages.append({"role": "assistant", "content": res.content})
