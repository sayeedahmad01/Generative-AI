import streamlit as st
from transformers import pipeline

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="GPT-Neo Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stChatMessage {
    border-radius: 10px;
    padding: 10px;
}

h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚙️ Settings")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=1.5,
    value=0.8,
    step=0.1
)

max_length = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=500,
    value=150,
    step=10
)

st.sidebar.markdown("---")
st.sidebar.info("GPT-Neo Language Model")

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model="distilgpt2"
    )
    return generator

generator = load_model()

# -------------------------
# Title
# -------------------------
st.title("🤖 GPT-Neo Chat Assistant")
st.caption("AI-Powered Text Generation Dashboard")

# -------------------------
# Session State
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Show Chat History
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# User Input
# -------------------------
prompt = st.chat_input("Ask me anything...")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True
            )

            answer = response[0]["generated_text"]

            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "Developed by **Sayeed Ahmad** | GPT-Neo Language Model"
)
