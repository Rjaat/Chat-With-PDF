import streamlit as st 
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from streamlit_chat import message
import streamlit_lottie as st_lottie
import requests

# Set page configuration
st.set_page_config(
    page_title="ChatPDF Pro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to load Lottie animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load Lottie animations
lottie_pdf = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fpbn0m4z.json")
lottie_chat = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_csn3mwcc.json")

checkpoint = "MBZUAI/LaMini-Flan-T5-783M" #"MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

persist_directory = "db"

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    db=None 

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p = 0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def main():
    st.markdown("<h1 style='text-align: center; color: #007bff;'>ChatPDF Pro 📚</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>Unlock the power of your PDFs with AI-driven conversations</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    with col1:
        if lottie_pdf:
            st_lottie.st_lottie(lottie_pdf, height=200, key="pdf_animation")
        else:
            st.image("https://www.iconpacks.net/icons/2/free-pdf-upload-icon-2617-thumb.png", width=200)

    with col2:
        st.markdown("<h2 style='color: #007bff;'>Upload your PDF</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{get_file_size(uploaded_file) / 1024:.2f} KB"
        }
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown("<h4 style='color: #007bff;'>File Details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style='color: #007bff;'>File Preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('Creating embeddings... This may take a moment.'):
                ingested_data = data_ingestion()
            st.success('Embeddings created successfully!')
            st.markdown("<h4 style='color: #007bff;'>Chat with your PDF</h4>", unsafe_allow_html=True)
            if lottie_chat:
                st_lottie.st_lottie(lottie_chat, height=100, key="chat_animation")
            else:
                st.image("https://www.iconpacks.net/icons/2/free-chat-icon-2639-thumb.png", width=100)

            user_input = st.text_input("Ask a question about your PDF:", key="input")

            if "generated" not in st.session_state:
                st.session_state["generated"] = ["Welcome! I'm ready to help you explore your PDF. What would you like to know?"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hello!"]
                
            if user_input:
                with st.spinner('Analyzing your PDF...'):
                    answer = process_answer({'query': user_input})
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            if st.session_state["generated"]:
                st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
                display_conversation(st.session_state)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
    """
    <div style='border-top: 2px solid #007bff; margin-top: 30px;'></div>
    <div style='text-align: center; padding-top: 20px; color: #6c757d; font-family: "Helvetica Neue", sans-serif;'>
        <p style='font-size: 16px; font-weight: 400;'>
            &copy; 2024. All rights reserved.
        </p>
        <p style='font-size: 16px; font-weight: 300; margin-bottom: 10px;'>
            Connect with me:
        </p>
        <p style='font-size: 20px; font-weight: 500;'>
            <a href='https://www.linkedin.com/in/rajesh-choudharyy/' style='text-decoration: none; color: #007bff; margin: 0 15px;' target='_blank'>
                <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' alt='LinkedIn' width='28' height='28' style='vertical-align: middle;'>
                <span style='margin-left: 8px;'>LinkedIn</span>
            </a>
            <a href='https://github.com/Rjaat' style='text-decoration: none; color: #007bff; margin: 0 15px;' target='_blank'>
                <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' alt='GitHub' width='28' height='28' style='vertical-align: middle;'>
                <span style='margin-left: 8px;'>GitHub</span>
            </a>
            <a href='https://rjaat.github.io/' style='text-decoration: none; color: #007bff; margin: 0 15px;' target='_blank'>
                <img src='https://cdn-icons-png.flaticon.com/512/1170/1170576.png' alt='Portfolio' width='28' height='28' style='vertical-align: middle;'>
                <span style='margin-left: 8px;'>Portfolio</span>
            </a>
        </p>
    </div>
    <div style='border-top: 2px solid #007bff; margin-top: 20px;'></div>
    """,
    unsafe_allow_html=True
)




if __name__ == "__main__":
    main()