from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


# -------- Load PDF --------
def load_pdf(file):

    pdf_reader = PdfReader(file)

    text = ""

    for page in pdf_reader.pages:

        content = page.extract_text()

        if content:
            text += content + "\n"

    return text


# -------- Chunk Text --------
def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(text)

    return chunks


# -------- Create Embeddings --------
def create_embeddings():

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings_model


# -------- Store in FAISS --------
def store_in_faiss(chunks, embeddings_model):

    vector_db = FAISS.from_texts(
        chunks,
        embedding=embeddings_model
    )

    return vector_db