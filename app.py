from flask import Flask, render_template, request, session
import os

from services.pdf_service import extract_pdf_text
from services.ranking_service import get_important_chunks
from services.embedding_service import (
    create_embeddings,
    search_chunks,
    add_sentences_to_chunks
)
from services.query_service import (
    detect_query_intent,
    expand_query
)
from services.llm_service import generate_answer
from services.rerank_service import rerank_results

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-later"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DOCUMENT_CACHE = {}


def get_file_cache_key(path):
    stat = os.stat(path)
    return f"{path}-{stat.st_mtime}-{stat.st_size}"


def process_document(path):
    cache_key = get_file_cache_key(path)

    if cache_key in DOCUMENT_CACHE:
        return DOCUMENT_CACHE[cache_key]

    pages, chunks = extract_pdf_text(path)

    chunks = add_sentences_to_chunks(chunks)

    important_chunks = get_important_chunks(chunks)

    embeddings = create_embeddings(chunks)

    DOCUMENT_CACHE[cache_key] = {
        "pages": pages,
        "chunks": chunks,
        "important_chunks": important_chunks,
        "embeddings": embeddings
    }

    return DOCUMENT_CACHE[cache_key]


@app.route("/", methods=["GET", "POST"])
def index():

    filename = None

    pages = []
    chunks = []

    important_chunks = []

    semantic_results = []

    ai_answer = ""

    query = ""
    query_intent = ""

    if request.method == "POST":

        query = request.form.get("query", "")

        file = request.files.get("pdf")

        if file and file.filename.endswith(".pdf"):

            filename = file.filename

            path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(path)

            session["filename"] = filename

        filename = session.get("filename")

        if filename:

            path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            if os.path.exists(path):

                document_data = process_document(path)

                pages = document_data["pages"]

                chunks = document_data["chunks"]

                important_chunks = document_data["important_chunks"]

                embeddings = document_data["embeddings"]

                if not query:
                    query = "vad är viktigast i dokumentet"

                query_intent = detect_query_intent(query)

                search_query = expand_query(query)

                if query_intent == "location":

                    semantic_results = search_chunks(
                        search_query,
                        chunks,
                        embeddings,
                        limit=10,
                        min_score=0.20,
                        keyword_weight=1.0,
                        semantic_weight=0.15,
                        query_intent=query_intent
                    )

                else:

                    semantic_results = search_chunks(
                        search_query,
                        chunks,
                        embeddings,
                        limit=10,
                        min_score=0.25,
                        keyword_weight=0.45,
                        semantic_weight=0.55,
                        query_intent=query_intent
                    )

                if semantic_results:

                    semantic_results = rerank_results(
                        query,
                        semantic_results,
                        limit=6
                    )

                    ai_answer = generate_answer(
                        query,
                        semantic_results
                    )

    return render_template(
        "index.html",
        filename=filename,
        pages=pages,
        chunks=chunks,
        important_chunks=important_chunks,
        semantic_results=semantic_results,
        ai_answer=ai_answer,
        query=query,
        query_intent=query_intent
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)