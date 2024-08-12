# ChatPDF Pro 📚

ChatPDF Pro is an AI-powered application that enables users to interact with their PDF documents through a chat interface. By leveraging natural language processing (NLP) models, it allows users to ask questions about their PDFs and receive insightful answers in real-time. The project is built using Streamlit and LangChain, with integrations from Hugging Face and other state-of-the-art NLP tools.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features

- **Interactive Chat Interface:** Ask questions about your PDFs and receive AI-generated responses.
- **PDF Ingestion:** Upload and process PDF files to extract and analyze content.
- **Embeddings Creation:** Create text embeddings using the `all-MiniLM-L6-v2` model for efficient text retrieval.
- **Lottie Animations:** Visually appealing Lottie animations integrated into the app interface.
- **Streamlit Integration:** A user-friendly web interface built using Streamlit.
  
![Screenshot from 2024-08-12 16-02-25](https://github.com/user-attachments/assets/1b39bbdc-386e-4a3e-bd7f-4e6ba4b5fab0)


## Installation

To set up ChatPDF Pro locally, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Rjaat/Chat-With-PDF.git
   cd Chat-With-PDF
2. **Create a Virtual Environment:**

     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
3.  **Install Dependencies:**
     ```
     pip install -r requirements.txt
     ```
4. **Download the Required Models:**
     The project uses the MBZUAI/LaMini-T5-738M model. You can download it using the following:
     ```
     from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
     checkpoint = "MBZUAI/LaMini-T5-738M"
     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
     model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
     ```
## Usage
1. ***Run the Streamlit App:***
     ```
     streamlit run app.py
     ```
2. ***Upload a PDF File:***
     - Navigate to the app in your browser.
     - Upload your PDF file using the file uploader in the interface.
3. ***Interact with Your PDF:***
     - After the file is processed, you can start asking questions related to the content of your PDF.

## Project Structure:
```
.
├── app.py                       # Main application script
├── docs/                        # Directory for storing uploaded PDF files
├── db/                          # Directory for storing embeddings database
├── requirements.txt             # Python dependencies
├── README.md                    # Project README
```
     
