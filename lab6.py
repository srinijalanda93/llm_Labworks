import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_models():
    english_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    hindi_model = pipeline("question-answering", model="ai4bharat/indic-bert")
    telugu_model = pipeline("question-answering", model="l3cube-pune/telugu-question-answering-squad-bert")
    return english_model, hindi_model, telugu_model

st.title("🧠 Logical Question Answering: English, Hindi, Telugu")
st.markdown("Supports logical inference-based questions like syllogism and blood relation problems.")

language = st.selectbox(" Select Language", ["English", "Hindi", "Telugu"])
question = st.text_area("❓ Enter Logical Question")
context = st.text_area("📘 Provide Relevant Context or Premise")

if st.button("🔍 Get Answer"):
    if not question or not context:
        st.warning("🚫 Please enter both the question and the context.")
    else:
        english_qa, hindi_qa, telugu_qa = load_models()

        if language == "English":
            model = english_qa
        elif language == "Hindi":
            model = hindi_qa
        else:  # Telugu
            model = telugu_qa

        try:
            result = model(question=question, context=context)
            st.success(f"✅ Answer: {result['answer']}")
            st.info(f"🧪 Confidence Score: {result['score']:.4f}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

