import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def build_evidence_context(evidence_chunks):
    context_parts = []

    for index, chunk in enumerate(evidence_chunks, start=1):

        evidence_blocks = chunk.get("display_evidence", [])

        if not evidence_blocks:
            continue

        evidence_text = " ".join(evidence_blocks)

        context_parts.append(
            f"Stöd {index} (sida {chunk['page']}): {evidence_text}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query, evidence_chunks):

    context = build_evidence_context(evidence_chunks)

    prompt = f"""
Du är en försiktig dokumentassistent.

Du får bara använda evidensen nedan.
Du får inte lägga till egen kunskap.
Om evidensen inte räcker ska du skriva:
"Det framgår inte tydligt av dokumentet."

Fråga:
{query}

Evidens:
{context}

Skriv ett kort, naturligt svar på svenska.

Regler:
- Sammanfatta evidensen istället för att kopiera den ord för ord.
- Skriv helst i vanlig brödtext, inte punktlista.
- Om flera saker nämns, väv ihop dem i 2–4 korta meningar.
- Använd inte markdown, listpunkter, stjärnor eller numrering.
- Skriv professionellt och tydligt.
- Max 4 meningar.
- Nämn inte "Stöd 1" eller interna scores i svaret.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 180
            }
        }
    )

    data = response.json()

    return data["response"].strip()