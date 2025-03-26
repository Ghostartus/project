from flask import Flask, render_template, request
import faiss
import google.generativeai as genai
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

genai.configure(api_key="AIzaSyArr8gWELi_PzeXLiFgdBB8_UeVIewpbWw")

# ✅ Load Sentence Transformer model
embedding_model = SentenceTransformer("intfloat/e5-large-v2")

# ✅ Predefined PDF file
PDF_PATH = "document.pdf"

# ✅ Step 1: Extract text from predefined PDF
def extract_text_from_pdf():
    with pdfplumber.open(PDF_PATH) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text.strip()

# ✅ Step 2: Convert text to vector embeddings & store in FAISS
def store_text_in_faiss(text):
    sections = text.split("\n\n")
    sections = [s.strip() for s in sections if len(s) > 50]

    embeddings = embedding_model.encode(sections, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])  # ✅ Changed variable name
    faiss_index.add(embeddings)

    return faiss_index, sections, embeddings  # ✅ Renamed to avoid conflict

# ✅ Step 3: Retrieve best-matching context from FAISS
def retrieve_relevant_text(query, faiss_index, sections):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    scores, idx = faiss_index.search(query_embedding, k=5)  # ✅ Using correct FAISS index

    best_matches = [sections[i] for i in idx[0] if scores[0][i] < 5.0]
    return "\n\n".join(best_matches[:3]) if best_matches else "No relevant section found."

# ✅ Step 4: Ask Gemini AI with context
def ask_ai(query, context):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")
    return response.text if hasattr(response, "text") else "No answer generated."

# ✅ Load & index the PDF when the server starts
pdf_text = extract_text_from_pdf()
faiss_index, sections, embeddings = store_text_in_faiss(pdf_text)  # ✅ Avoiding name conflicts

@app.route("/", methods=["GET", "POST"])
def home():  # ✅ Renamed function to avoid conflicts
    answer = None
    question = None

    if request.method == "POST":
        question = request.form["question"]
        context = retrieve_relevant_text(question, faiss_index, sections)  # ✅ Using correct index
        answer = ask_ai(question, context)

    return render_template("index.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
