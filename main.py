import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="GenAI Text Generator",
    page_icon="🤖"
)


st.title("🤖 GenAI Text Generator")


@st.cache_resource
def load_model():

    model = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2"
    )

    return model


with st.spinner("Loading AI model..."):
    generator = load_model()


prompt = st.text_area(
    "Enter your prompt"
)


if st.button("Generate"):

    if prompt.strip():

        output = generator(
            prompt,
            max_length=150,
            num_return_sequences=1
        )

        st.subheader("Generated Text")

        st.write(
            output[0]["generated_text"]
        )

    else:
        st.warning("Enter some text first")
