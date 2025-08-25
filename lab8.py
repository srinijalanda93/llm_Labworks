import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import os
import tempfile
from googletrans import Translator
import torch

# Domain: AI and Machine Learning (sample knowledge base)
documents = [
    "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "Machine Learning (ML) is a subset of AI that enables systems to learn from data.",
    "Deep Learning is a subset of ML using neural networks with many layers.",
    "Natural Language Processing (NLP) is a field of AI focused on interaction between computers and humans using natural language.",
    "Computer Vision allows computers to interpret and understand the visual world.",
    "Reinforcement Learning is a type of ML where agents learn by interacting with an environment.",
    "Supervised Learning uses labeled data to train models.",
    "Unsupervised Learning finds patterns in unlabeled data.",
    "Generative AI creates new content, like images or text, based on training data.",
    "xAI is a company founded by Elon Musk to understand the universe."
]

# Custom embedding class for SentenceTransformers, inheriting from Embeddings
class SentenceTransformerEmbeddings(Embeddings):
    def _init_(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

# Function to create or load vector store
@st.cache_resource
def create_vectorstore():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save documents to temporary files
        for i, doc in enumerate(documents):
            file_path = os.path.join(temp_dir, f"doc_{i}.txt")
            with open(file_path, "w") as f:
                f.write(doc)
        
        # Load documents
        docs = []
        for i in range(len(documents)):
            file_path = os.path.join(temp_dir, f"doc_{i}.txt")
            loader = TextLoader(file_path)
            docs.extend(loader.load())
        
        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(docs)
        
        # Embeddings
        embeddings = SentenceTransformerEmbeddings()
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore

# Initialize Phi-3-mini-4k-instruct model (lightweight, ungated)
@st.cache_resource
def load_llm():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Reduce memory usage
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=pipe)

# Custom prompt for RAG
prompt_template = """You are an expert in AI and Machine Learning. Use the following context to answer the question accurately.
If you don't know the answer, say so.

Context: {context}

Question: {question}

Answer in the language of the question if possible."""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create RetrievalQA chain
vectorstore = create_vectorstore()
llm = load_llm()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT}
)

# Multilingual support
translator = Translator()

def translate_text(text, dest_lang):
    try:
        return translator.translate(text, dest=dest_lang).text
    except:
        return text

def detect_language(query):
    try:
        return translator.detect(query).lang
    except:
        return 'en'

# Streamlit Interface
st.title("AI & ML Domain-Specific Chatbot")
st.write("Ask questions about AI and Machine Learning. Supports multilingual queries (e.g., English, Spanish, French).")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Detect query language
    query_lang = detect_language(prompt)
    
    # Translate to English for RAG if needed
    if query_lang != 'en':
        english_prompt = translate_text(prompt, 'en')
    else:
        english_prompt = prompt
    
    # Get response from RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.run(english_prompt)
            
            # Translate response back to query language if needed
            if query_lang != 'en':
                response = translate_text(response, query_lang)
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})