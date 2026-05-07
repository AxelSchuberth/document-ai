import fitz
import re


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def create_sentence_chunks(all_sentences, max_words=220):
    chunks = []
    current_sentences = []
    current_word_count = 0
    chunk_id = 1
    start_page = None

    for item in all_sentences:
        sentence = item["sentence"]
        page = item["page"]
        word_count = len(sentence.split())

        if start_page is None:
            start_page = page

        if current_word_count + word_count > max_words and current_sentences:
            chunk_text = " ".join(current_sentences)

            chunks.append({
                "id": chunk_id,
                "page": start_page,
                "text": chunk_text
            })

            chunk_id += 1
            current_sentences = []
            current_word_count = 0
            start_page = page

        current_sentences.append(sentence)
        current_word_count += word_count

    if current_sentences:
        chunk_text = " ".join(current_sentences)

        chunks.append({
            "id": chunk_id,
            "page": start_page,
            "text": chunk_text
        })

    return chunks


def extract_pdf_text(file_path):
    doc = fitz.open(file_path)

    pages = []
    all_sentences = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        text = page.get_text("text")

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = " ".join(lines)

        pages.append({
            "page": page_num + 1,
            "text": cleaned_text
        })

        sentences = split_into_sentences(cleaned_text)

        for sentence in sentences:
            all_sentences.append({
                "sentence": sentence,
                "page": page_num + 1
            })

    chunks = create_sentence_chunks(all_sentences, max_words=220)

    return pages, chunks