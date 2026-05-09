import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2:3b"


def generate_answer(query, evidence_chunks):

    context_parts = []

    for chunk in evidence_chunks:
        context_parts.append(chunk["text"])

    context = "\n\n".join(context_parts)

    prompt = f"""
    Du är en intelligent dokumentassistent.

    Uppgift:
    - Besvara frågan kort och naturligt på svenska.
    - Använd ENDAST information från evidensen.
    - Sammanfatta istället för att kopiera meningar.
    - Om informationen inte räcker, säg:
    "Det framgår inte tydligt av dokumentet."

    Fråga:
    {query}

    Evidens:
    {context}

    Regler:
    - Skriv professionellt och tydligt.
    - Skriv som en riktig AI-assistent.
    - Upprepa inte evidensen ord för ord.
    - Max 4 meningar.
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    return data["response"].strip()