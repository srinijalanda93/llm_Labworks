import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import time

# Configuration
st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("LAB-3 Comparative Analysis of Foundation vs Domain-Specific Models")

# Model loading without pipelines
@st.cache_resource
def load_models():
    models = {}
    
    # Foundation Model - GPT-2
    models['gpt2'] = {
        'tokenizer': AutoTokenizer.from_pretrained('gpt2'),
        'model': AutoModelForCausalLM.from_pretrained('gpt2'),
        'type': 'generation'
    }
    models['gpt2']['tokenizer'].pad_token = models['gpt2']['tokenizer'].eos_token
    
    # Finance Model - FinBERT
    models['finbert'] = {
        'tokenizer': AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone'),
        'model': AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone'),
        'type': 'classification'
    }
    
    # Healthcare Model - BioBERT
    models['biobert'] = {
        'tokenizer': AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1'),
        'model': AutoModelForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1'),
        'type': 'classification'
    }
    
    return models

models = load_models()

# Helper functions
def generate_text(model_info, prompt, max_length=100):
    inputs = model_info['tokenizer'](prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_info['model'].generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=model_info['tokenizer'].eos_token_id
        )
    return model_info['tokenizer'].decode(outputs[0], skip_special_tokens=True)

def classify_text(model_info, text):
    inputs = model_info['tokenizer'](text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_info['model'](**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()[0]

# Interface
st.sidebar.header("Configuration")
task_type = st.sidebar.selectbox(
    "Select Task Type",
    ["Text Generation", "Text Classification"]
)

max_length = st.sidebar.slider("Max output length", 50, 500, 100)

# Main comparison
st.header("Model Comparison")

input_text = st.text_area("Input text for analysis:", 
                         "The patient showed symptoms of myocardial infarction and was admitted to the cardiac unit.")

if st.button("Run Comparison"):
    results = defaultdict(dict)
    timing = {}
    
    # Run each model
    for model_name, model_info in models.items():
        start_time = time.time()
        
        if model_info['type'] == 'generation':
            output = generate_text(model_info, input_text, max_length)
            results[model_name]['output'] = output
        else:
            probs = classify_text(model_info, input_text)
            results[model_name]['probabilities'] = probs
        
        timing[model_name] = time.time() - start_time
    
    # Display results
    cols = st.columns(len(models))
    
    for idx, (model_name, result) in enumerate(results.items()):
        with cols[idx]:
            st.subheader(model_name.upper())
            
            if models[model_name]['type'] == 'generation':
                st.text_area("Generated Output:", 
                            result['output'], 
                            height=200)
            else:
                st.write("Classification Probabilities:")
                probs = result['probabilities']
                if model_name == 'finbert':
                    labels = ['Negative', 'Neutral', 'Positive']
                else:
                    labels = [f"Class {i}" for i in range(len(probs))]
                
                prob_df = pd.DataFrame({
                    'Label': labels,
                    'Probability': probs
                })
                st.bar_chart(prob_df.set_index('Label'))
            
            st.caption(f"Execution time: {timing[model_name]:.2f}s")
    
    # Qualitative comparison
    st.header("Qualitative Analysis")
    
    comparison_data = {
        "Model": list(models.keys()),
        "Relevance": [0.7, 0.9, 0.8],  # These would be filled based on actual evaluation
        "Domain Understanding": [0.6, 0.95, 0.9],
        "Speed (seconds)": list(timing.values())
    }
    
    st.write("Relative Performance Characteristics (Hypothetical Example)")
    st.bar_chart(pd.DataFrame(comparison_data).set_index('Model'))

# Model information
with st.expander("Model Details"):
    st.markdown("""
    ### Model Specifications:
    
    - **GPT-2 (Foundation Model)**
      - Parameters: 117M
      - Training Data: Diverse internet text
      - Best for: General text generation
    
    - **FinBERT (Financial Domain)**
      - Parameters: 110M
      - Training Data: Financial news, reports
      - Best for: Financial sentiment analysis
    
    - **BioBERT (Healthcare Domain)**
      - Parameters: 110M
      - Training Data: Biomedical literature
      - Best for: Medical text classification
    """)

# Important disclaimer
st.warning("""
**Note:** The performance metrics shown are for demonstration purposes only. 
Actual evaluation would require proper benchmarking on domain-specific datasets.
""")