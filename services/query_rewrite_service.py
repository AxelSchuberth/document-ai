import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def rewrite_query_for_search(query):
    prompt = f"""
Du hjälper ett dokument-söksystem att hitta bättre evidens.

Skriv om användarens fråga till korta svenska söktermer.
Returnera ENDAST söktermerna.
Ingen förklaring.
Ingen punktlista.
Ingen markdown.

Exempel:
Fråga: Vad har vi lärt oss?
Söktermer: lärdomar reflektion erfarenheter insikter vad vi tar med oss

Fråga: Vilka risker finns?
Söktermer: risker problem utmaningar hinder hot svårigheter

Fråga:
{query}

Söktermer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 60
            }
        }
    )

    data = response.json()

    return data["response"].strip()