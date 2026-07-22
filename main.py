import streamlit as st
from transformers import pipeline
import torch


# Page Configuration
st.set_page_config(
    page_title="GenAI Text Generator",
    page_icon="🤖",
    layout="centered"
)


# Title
st.title("🤖 GenAI Text Generator")
st.write(
    "Generate human-like text using Transformer based AI model"
)


# Load Model
@st.cache_resource
def load_model():

    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if torch.cuda.is_available() else -1
    )

    return generator


generator = load_model()


# User Input

prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Example: Explain Artificial Intelligence in simple words"
)


# Parameters

col1, col2 = st.columns(2)


with col1:
    max_length = st.slider(
        "Maximum Length",
        50,
        300,
        150
    )


with col2:
    temperature = st.slider(
        "Creativity Level",
        0.1,
        1.5,
        0.7
    )


# Generate Button

if st.button("🚀 Generate Text"):

    if prompt.strip() == "":
        st.warning(
            "Please enter some text"
        )

    else:

        with st.spinner("Generating response..."):

            result = generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1
            )


            generated_text = result[0]["generated_text"]


            st.subheader(
                "Generated Output"
            )

            st.success(
                generated_text
            )


# Footer

st.divider()

st.caption(
    "Built with Streamlit + Hugging Face Transformers + GPT-2"
)
