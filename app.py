import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="MANHATTAN", page_icon="🤖")
st.title("MANHATTAN 🤖")

api_key = st.secrets["OPENROUTER_API_KEY"]

llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are MANHATTAN, a helpful AI chatbot made by Genpachino. Never say you are ERNIE Bot or Baidu. Always say you are MANHATTAN."},
        {"role": "assistant", "content": "HELLO! I am your best companion MANHATTAN"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Kehi sodha..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            all_msgs = []
            for m in st.session_state.messages:
                if m["role"] == "system":
                    all_msgs.append(SystemMessage(content=m["content"]))
                elif m["role"] == "user":
                    all_msgs.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    all_msgs.append(AIMessage(content=m["content"]))
            res = llm.invoke(all_msgs)
            st.session_state.messages.append({"role": "assistant", "content": res.content})
            st.write(res.content)
