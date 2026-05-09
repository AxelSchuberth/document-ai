import re


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]