"""
University Q&A Chatbot — Streamlit UI

Run with:
    export OPENAI_API_KEY="your-key-here"
    streamlit run app.py

Make sure you've run `python build_index.py` at least once before starting this.

"""

import streamlit as st
from rag_utils import get_client, load_index, answer_question

st.set_page_config(page_title="University Q&A Bot", page_icon="🎓")
st.title("🎓 University Q&A Assistant")
st.caption("Ask me anything about registration, GPA rules, graduation requirements, and more.")

# --- Load resources once, cache across reruns ---
@st.cache_resource
def load_resources():
    client = get_client()
    index, chunks = load_index()
    return client, index, chunks

try:
    client, index, chunks = load_resources()
except Exception as e:
    st.error(f"Could not load the index: {e}")
    st.stop()

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}:**\n\n{src}")

# --- Chat input ---
if prompt := st.chat_input("Ask a question about the university..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(prompt, client, index, chunks, k=3)
        st.markdown(answer)
        with st.expander("Sources used"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Source {i}:**\n\n{src}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )