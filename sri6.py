import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load all models and tokenizers
@st.cache_resource
def load_all_models():
    # English
    en_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Hindi & Telugu - use multilingual model
    indic_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    indic_model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")

    return en_model, (indic_model, indic_tokenizer)

# Indic language answer function
# def get_indic_answer(model, tokenizer, context, question):
#     inputs = tokenizer(question, context, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     start_idx = torch.argmax(outputs.start_logits)
#     end_idx = torch.argmax(outputs.end_logits) + 1
#     answer = tokenizer.convert_tokens_to_string(
#         tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
#     )
#     return answer.strip()

import torch.nn.functional as F

def get_indic_answer(model, tokenizer, context, question):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get start and end logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Get the most probable start and end positions
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits) + 1

    # Compute softmax probabilities
    start_probs = F.softmax(start_logits, dim=1)
    end_probs = F.softmax(end_logits, dim=1)

    # Confidence score = product of start and end probabilities
    confidence = (start_probs[0][start_idx] * end_probs[0][end_idx - 1]).item()

    # Decode the answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
    )
    return answer.strip(), confidence


# Streamlit UI
st.set_page_config(page_title="Logical QA - English, Hindi, Telugu")
st.title(" Question Answering System using 3 language")
st.markdown("Supports **Syllogism**, **Blood Relations**, and other logic types in **English, Hindi, and Telugu**.")

language = st.selectbox("🌐 Choose Language", ["English", "Hindi", "Telugu"])
question = st.text_area("❓ Enter Logical Question")
context = st.text_area("📘 Provide Context or Premise")

if st.button("🔍 Get Answer"):
    if not question or not context:
        st.warning("⚠️ Please enter both question and context.")
    else:
        with st.spinner("Thinking..."):
            en_model, (indic_model, indic_tokenizer) = load_all_models()

            try:
                if language == "English":
                    result = en_model(question=question, context=context)
                    st.success(f" Answer: {result['answer']}")
                    st.markdown(f" Confidence Score: {result['score']:.4f}")
                else:
                    answer, confidence = get_indic_answer(indic_model, indic_tokenizer, context, question)
                    st.success(f"Answer: {answer}")
                    st.markdown(f"Confidence Score: {confidence:.4f}")


            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
