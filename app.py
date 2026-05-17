from flask import Flask, render_template, request, session
import os
import time

from services.document_processing.pdf_service import extract_pdf_text
from services.retrieval.ranking_service import get_important_chunks

from services.retrieval.embedding_service import (
    create_embeddings,
    search_chunks,
    add_sentences_to_chunks,
    get_embedding_model
)

from services.queries.query_service import (
    detect_query_intent,
    expand_query
)

from services.ai.llm_service import generate_answer
from services.retrieval.rerank_service import rerank_results
from services.ai.query_rewrite_service import rewrite_query_for_search

from services.retrieval.sentence_search_service import (
    build_sentence_index,
    search_sentences,
    improve_results_with_sentence_hits
)

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-later"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DOCUMENT_CACHE = {}


def get_answer_confidence(results):
    if not results:
        return {
            "label": "Otillräckligt stöd",
            "class": "insufficient-confidence",
            "description": "Svaret saknar tydligt stöd i dokumentet."
        }

    top_result = results[0]
    high_count = sum(
        1 for result in results
        if result.get("confidence_level") == "high"
    )
    medium_count = sum(
        1 for result in results
        if result.get("confidence_level") == "medium"
    )
    evidence_count = sum(
        len(result.get("display_evidence", []))
        for result in results
    )
    top_score = top_result.get("final_score", 0)

    if high_count >= 2 and evidence_count >= 2 and top_score >= 0.45:
        return {
            "label": "Hög svarssäkerhet",
            "class": "high-confidence",
            "description": "Svaret bygger på flera tydliga stöd i dokumentet."
        }

    if high_count >= 1 or (medium_count >= 2 and evidence_count >= 2):
        return {
            "label": "Medel svarssäkerhet",
            "class": "medium-confidence",
            "description": "Svaret har stöd, men bör kontrolleras mot utdragen."
        }

    return {
        "label": "Låg svarssäkerhet",
        "class": "low-confidence",
        "description": "Svaret bygger på svagt eller begränsat stöd."
    }


def get_file_cache_key(path):
    stat = os.stat(path)
    return f"{path}-{stat.st_mtime}-{stat.st_size}"


def process_document(path):
    started_at = time.perf_counter()
    cache_key = get_file_cache_key(path)

    if cache_key in DOCUMENT_CACHE:
        print(f"Document cache hit: {time.perf_counter() - started_at:.2f}s")
        return DOCUMENT_CACHE[cache_key]

    pages, chunks = extract_pdf_text(path)

    chunks = add_sentences_to_chunks(chunks)

    important_chunks = get_important_chunks(chunks)

    DOCUMENT_CACHE[cache_key] = {
        "pages": pages,
        "chunks": chunks,
        "important_chunks": important_chunks,
        "embeddings": None,
        "sentence_items": None,
        "sentence_embeddings": None
    }

    print(
        "Document processed: "
        f"{len(pages)} pages, {len(chunks)} chunks, "
        f"{time.perf_counter() - started_at:.2f}s"
    )

    return DOCUMENT_CACHE[cache_key]


def ensure_document_embeddings(document_data):
    if document_data["embeddings"] is None:
        document_data["embeddings"] = create_embeddings(
            document_data["chunks"]
        )

    return document_data["embeddings"]


def ensure_sentence_index(document_data):
    if document_data["sentence_items"] is None:
        sentence_items, sentence_embeddings = build_sentence_index(
            document_data["chunks"],
            get_embedding_model()
        )

        document_data["sentence_items"] = sentence_items
        document_data["sentence_embeddings"] = sentence_embeddings

    return (
        document_data["sentence_items"],
        document_data["sentence_embeddings"]
    )


@app.route("/", methods=["GET", "POST"])
def index():

    filename = None

    pages = []
    chunks = []

    important_chunks = []

    semantic_results = []

    ai_answer = ""
    answer_confidence = None

    query = ""
    query_intent = ""

    rewritten_query = ""

    if request.method == "POST":

        query = request.form.get("query", "")

        file = request.files.get("pdf")
        uploaded_new_file = False

        if file and file.filename.endswith(".pdf"):

            filename = file.filename

            path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(path)

            session["filename"] = filename
            uploaded_new_file = True

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

                should_search = bool(query) or not uploaded_new_file

                if should_search and not query:
                    query = "vad är viktigast i dokumentet"

                if should_search:
                    embeddings = ensure_document_embeddings(document_data)

                    sentence_items, sentence_embeddings = ensure_sentence_index(
                        document_data
                    )

                    search_started_at = time.perf_counter()

                    query_intent = detect_query_intent(query)

                    search_query = expand_query(query)

                    # FÖRSTA RETRIEVAL

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

                    # SENTENCE-LEVEL SEARCH

                    sentence_results = search_sentences(
                        query,
                        sentence_items,
                        sentence_embeddings,
                        get_embedding_model(),
                        limit=12
                    )

                    # FÖRBÄTTRA CHUNK-EVIDENS MED SENTENCE-HITS

                    semantic_results = improve_results_with_sentence_hits(
                        semantic_results,
                        sentence_results,
                        max_sentences_per_chunk=3
                    )

                    # FALLBACK VIA OLLAMA QUERY REWRITE

                    if not semantic_results:

                        rewritten_query = rewrite_query_for_search(query)

                        fallback_query = expand_query(
                            query + " " + rewritten_query
                        )

                        semantic_results = search_chunks(
                            fallback_query,
                            chunks,
                            embeddings,
                            limit=10,
                            min_score=0.18,
                            keyword_weight=0.55,
                            semantic_weight=0.45,
                            query_intent=query_intent
                        )

                    # RERANKING

                    if semantic_results:

                        semantic_results = rerank_results(
                            query,
                            semantic_results,
                            limit=6
                        )

                        # AI-SVAR

                        ai_answer = generate_answer(
                            query,
                            semantic_results
                        )

                    answer_confidence = get_answer_confidence(semantic_results)

                    print(
                        "Search completed: "
                        f"{len(semantic_results)} results, "
                        f"{time.perf_counter() - search_started_at:.2f}s"
                    )
    if query and not ai_answer:
        ai_answer = "Det framgår inte tydligt av dokumentet."
        answer_confidence = get_answer_confidence(semantic_results)

    return render_template(
        "index.html",
        filename=filename,
        pages=pages,
        chunks=chunks,
        important_chunks=important_chunks,
        semantic_results=semantic_results,
        ai_answer=ai_answer,
        answer_confidence=answer_confidence,
        query=query,
        query_intent=query_intent,
        rewritten_query=rewritten_query
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)
