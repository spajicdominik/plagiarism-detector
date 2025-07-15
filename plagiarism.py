import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import textwrap

st.sidebar.title("🧠 Choose Model")

model_choice = st.sidebar.radio(
    "Select a model:",
    [
        "all-mpnet-base-v2 (general-purpose)",
        "allenai/specter (scientific)",
        "paraphrase-multilingual-MiniLM-L12-v2 (multilingual)"
    ]
)

@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

if "mpnet" in model_choice:
    model_name = "all-mpnet-base-v2"
elif "specter" in model_choice:
    model_name = "allenai/specter"
else:
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"

model = load_model(model_name)

if "specter" in model_name:
    st.sidebar.info("🔬 Best for research papers, abstracts, and scientific documents.")
elif "multilingual" in model_name:
    st.sidebar.info("🌐 Supports 50+ languages. Good for cross-language text comparison.")
else:
    st.sidebar.info("📝 General-purpose model for English text.")
    
st.sidebar.success(f"✅ Loaded model: {model_name}")

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def split_into_chunks(text, num_chunks=4):
    words = text.split()
    chunk_size = len(words) // num_chunks
    chunks = []
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = None if i == num_chunks - 1 else (i + 1) * chunk_size
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
    
    return chunks

st.title("Plagiarism Detector")

st.markdown("""
Enter a original text and a suspicious text you want to check for plagiarism. The system
will use a pretrained transformer model to calculate plagiarism possibility with cosine similarity.
""")

st.header("📘 Reference Texts (Multiple PDFs)")
ref_files = st.file_uploader("Upload one or more reference PDFs", type="pdf", accept_multiple_files=True)

reference_texts = []
ref_names = []

if ref_files:
    for file in ref_files:
        text = extract_text_from_pdf(file)
        if text:
            reference_texts.append(text)
            ref_names.append(file.name)

st.header("🔍 Suspicious Text (Paste or PDF)")

sus_file = st.file_uploader("📎 Upload suspicious PDF (optional)", type="pdf", key="sus")
sus_text_input = st.text_area("📝 Or paste suspicious text below", height=200)

suspicious_text = ""
if sus_file:
    suspicious_text = extract_text_from_pdf(sus_file)
    st.success("✅ Suspicious text loaded from PDF.")
elif sus_text_input.strip():
    suspicious_text = sus_text_input.strip()

threshold = st.slider("⚠️ Similarity Threshold", 0.7, 0.95, 0.85, step=0.01)

if st.button("🚨 Check for Plagiarism"):
    if not reference_texts or not suspicious_text:
        st.warning("Please upload reference text(s) and provide suspicious text (PDF or pasted).")
    else:
        st.write("🔄 Encoding texts...")
        ref_embeddings = model.encode(reference_texts)
        sus_embedding = model.encode([suspicious_text])[0]

        similarities = cosine_similarity([sus_embedding], ref_embeddings)[0]

        st.subheader("📊 Similarity Results")

        flagged = False
        for i, score in enumerate(similarities):
            st.write(f"📄 **{ref_names[i]}** — Similarity: `{score:.4f}`")
            if score > threshold:
                st.error("⚠️ High similarity — possible plagiarism.")
                flagged = True
            else:
                st.success("✅ Low similarity.")

        avg_score = similarities.mean()
        st.write(f"\n**📈 Average Similarity Across References:** `{avg_score:.4f}`")

        if avg_score > threshold * 0.75:
            st.warning("⚠️ The suspicious text may be a combination of multiple sources.")
            
        chunks = split_into_chunks(suspicious_text, num_chunks=4)
        chunk_embeddings = model.encode(chunks)

        st.subheader("🔍 Suspicious Chunks")

        suspicious_found = False

        for i, chunk_emb in enumerate(chunk_embeddings):
            sims = cosine_similarity([chunk_emb], ref_embeddings)[0]
            top_idx = sims.argmax()
            top_score = sims[top_idx]
            source = ref_names[top_idx]

        if top_score > threshold:
            suspicious_found = True
            st.markdown(f"### ⚠️ Chunk {i+1}: Similar to **{source}**")
            st.write(f"**Similarity Score:** `{top_score:.4f}`")
            with st.expander("📖 Show Chunk Text"):
                st.code(chunks[i], language="text")

        if not suspicious_found:
            st.success("✅ No suspicious chunks detected above the similarity threshold.")

        with st.expander("📖 Show Suspicious Text"):
            st.code(suspicious_text[:3000], language="text")
