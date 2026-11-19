        ("📚 Learn something", "Explain a difficult concept simply", "Explain a difficult concept simply."),
        ("✍️ Writing assistant", "Write, improve, or rewrite text", "Help me write or improve some text."),
        ("💻 Coding help", "Debug code or learn programming", "Help me with a coding problem."),
    ]
    cols = [col1, col2, col1, col2]
    for i, (title, desc, prompt_text) in enumerate(prompts):
        with cols[i]:
            st.markdown(f'<div class="prompt-card"><div class="prompt-title">{title}</div><div class="prompt-desc">{desc}</div></div>', unsafe_allow_html=True)
            if st.button(prompt_text, key=f"suggest{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

prompt = st.chat_input("Message MANHATTAN...")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    all_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            all_msgs.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            all_msgs.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("MANHATTAN is thinking..."):
            try:
                api_key = st.secrets["OPENROUTER_API_KEY"]
                llm = ChatOpenAI(
                    model=model.strip(),
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
                st.error("Couldn't connect to OpenRouter. Check your API key, model ID, connection, or rate limit.")
                st.caption(f"Details: {e}")

st.markdown('<div class="footer">MANHATTAN AI · Powered by OpenRouter</div>', unsafe_allow_html=True)
