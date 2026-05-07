from flask import Flask, render_template, request
import os

from services.pdf_service import extract_pdf_text
from services.ranking_service import get_important_chunks
from services.embedding_service import (
    create_embeddings,
    search_chunks,
    add_sentences_to_chunks
)
from services.query_service import detect_query_intent

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    filename = None

    pages = []
    chunks = []

    important_chunks = []

    semantic_results = []
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

            with open("last_uploaded.txt", "w", encoding="utf-8") as f:
                f.write(filename)

        if os.path.exists("last_uploaded.txt"):

            with open("last_uploaded.txt", "r", encoding="utf-8") as f:
                filename = f.read().strip()

            path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            if os.path.exists(path):

                pages, chunks = extract_pdf_text(path)

                chunks = add_sentences_to_chunks(chunks)

                important_chunks = get_important_chunks(chunks)

                embeddings = create_embeddings(chunks)

                if not query:
                    query = "vad är viktigast i dokumentet"

                query_intent = detect_query_intent(query)

                if query_intent == "location":
                    semantic_results = search_chunks(
                        query,
                        chunks,
                        embeddings,
                        limit=5,
                        min_score=0.20,
                        keyword_weight=1.0,
                        semantic_weight=0.15,
                        query_intent=query_intent
                    )

                else:
                    semantic_results = search_chunks(
                        query,
                        chunks,
                        embeddings,
                        limit=5,
                        min_score=0.25,
                        keyword_weight=0.45,
                        semantic_weight=0.55,
                        query_intent=query_intent
                    )

    return render_template(
        "index.html",
        filename=filename,
        pages=pages,
        chunks=chunks,
        important_chunks=important_chunks,
        semantic_results=semantic_results,
        query=query,
        query_intent=query_intent
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)