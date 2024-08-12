from langchain_community.document_loaders import PyPDFLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
# from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    all_documents = []

    # Walk through the docs directory and process PDF files
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Processing file: {file}")
                file_path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    if documents:
                        all_documents.extend(documents)
                    else:
                        print(f"No documents found in {file}.")
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    if not all_documents:
        print("No documents were loaded.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)

    print("Loading sentence transformers model...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating embeddings. This may take some time...")
    # db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    db = None

    print("Ingestion complete! You can now run privateGPT.py to query your documents.")

if __name__ == "__main__":
    main()
