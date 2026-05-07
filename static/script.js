let activeChunkId = null;
let activeSentences = [];

function clearHighlights() {
    document.querySelectorAll(".highlight").forEach(element => {
        element.classList.remove("highlight");
    });
}

function normalizeText(text) {
    return text
        .replace(/\s+/g, " ")
        .replace(/[“”]/g, '"')
        .replace(/[‘’]/g, "'")
        .trim()
        .toLowerCase();
}

function scrollToSentences(chunkId, sentences) {
    const sameChunk = activeChunkId === chunkId;
    const sameSentences =
        JSON.stringify(activeSentences) === JSON.stringify(sentences);

    if (sameChunk && sameSentences) {
        clearHighlights();
        activeChunkId = null;
        activeSentences = [];
        return;
    }

    clearHighlights();

    activeChunkId = chunkId;
    activeSentences = sentences;

    const chunkElement = document.getElementById(chunkId);
    if (!chunkElement) return;

    chunkElement.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

    const sentenceElements = chunkElement.querySelectorAll(".sentence");

    const normalizedTargets = sentences.map(sentence =>
        normalizeText(sentence)
    );

    let foundAny = false;

    sentenceElements.forEach(sentenceElement => {
        const normalizedSentence = normalizeText(sentenceElement.innerText);

        normalizedTargets.forEach(target => {
            if (
                target &&
                (
                    normalizedSentence === target ||
                    normalizedSentence.includes(target) ||
                    target.includes(normalizedSentence)
                )
            ) {
                sentenceElement.classList.add("highlight");
                foundAny = true;
            }
        });
    });

    if (!foundAny) {
        chunkElement.classList.add("highlight");
    }
}