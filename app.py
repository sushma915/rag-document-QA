import streamlit as st
from sentence_transformers import CrossEncoder

from helper_functions import load_pdf, chunk_text, create_embeddings, store_in_faiss


# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Document QA System", page_icon="📄")

st.title("📄 Document Question Answering System")


# ---------------- Custom UI Style ----------------
st.markdown("""
<style>

.stApp{
background-color:#0f172a;
color:#f8fafc;
}

.chunk-card{
background-color:#1e293b;
padding:15px;
border-radius:10px;
margin-bottom:10px;
border:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)


# ---------------- Session State ----------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None


# ---------------- Load Cross Encoder ----------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------- Upload PDF ----------------
uploaded_files = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=True
)


if uploaded_files:

    pdf_texts = {}

    for file in uploaded_files:
        pdf_texts[file.name] = load_pdf(file)

    pdf_names = list(pdf_texts.keys())

    selected_pdf = st.selectbox("Select Document", pdf_names)

    text = pdf_texts[selected_pdf]


    # ---------------- Process PDF ----------------
    if st.session_state.current_pdf != selected_pdf:

        chunks = chunk_text(text)

        st.success(f"Document split into {len(chunks)} chunks")

        embeddings_model = create_embeddings()

        vector_db = store_in_faiss(chunks, embeddings_model)

        st.session_state.vector_db = vector_db
        st.session_state.current_pdf = selected_pdf

        st.success("Embeddings stored in FAISS")

    else:

        vector_db = st.session_state.vector_db


    # ---------------- User Query ----------------
    query = st.text_input("Enter your question")


    if query:

        docs_and_scores = vector_db.similarity_search_with_score(query, k=5)

        relevant_docs = [(doc, score) for doc, score in docs_and_scores if score < 1.0]

        if not relevant_docs:

            st.warning("No relevant chunks found")

        else:

            # ---------------- Cross Encoder Reranking ----------------
            pairs = [(query, doc.page_content) for doc, score in relevant_docs]

            rerank_scores = cross_encoder.predict(pairs)

            reranked_docs = sorted(
                zip(relevant_docs, rerank_scores),
                key=lambda x: x[1],
                reverse=True
            )


            # ---------------- Source Chunks ----------------
            st.subheader("Source Chunks")


            for i, ((doc, score), rerank_score) in enumerate(reranked_docs[:3], 1):

                st.markdown(f"""
<div class="chunk-card">

<b>Chunk {i}</b>

{doc.page_content}

<br>

<i>FAISS Score: {score:.4f}</i>

<br>

<i>CrossEncoder Score: {rerank_score:.4f}</i>

</div>
""", unsafe_allow_html=True)