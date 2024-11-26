import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredURLLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from dotenv import load_dotenv

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Create the Hugging Face pipeline using the 'pipeline' function
hf_pipeline = pipeline(
    "text2text-generation", # Use 'text2text-generation' for sequence-to-sequence models like T5
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"max_new_tokens": 512}
)

# Create the LangChain HuggingFacePipeline, passing the hf_pipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

st.title("News Research Chatbot")
st.sidebar.title("News Articles Links")

# Collecting URLs
urls = []

for i in range(3):
    url = st.sidebar.text_input(label=f"URL {i+1}", key=f"URL{i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# Initialize the placeholder
main_placeholder = st.empty()

if process_url_clicked:
    # Loading the data and updating the placeholder
    main_placeholder.write("Loading data...")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Splitting the data and updating the placeholder
    main_placeholder.write("Splitting data...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    texts = text_splitter.split_documents(data)

    # Creating embeddings and updating the placeholder
    main_placeholder.write("Creating embedding vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"trust_remote_code": True}
    )

    # Creating vector store and updating the placeholder
    db = FAISS.from_documents(texts, embeddings)
    main_placeholder.write("Embedding vector store created successfully!")

    # Saving vector store and updating the placeholder
    main_placeholder.write("Saving vector store...")
    time.sleep(2)  # Simulate processing time
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(db, f)
    
    main_placeholder.write("Vector store saved successfully!")



query = main_placeholder.text_input("Question:")
if query:
    with open("vectorstore.pkl", "rb") as f:
        db = pickle.load(f)
        chain  = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = db.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.subheader(result['answer'])

        


