let activeChunkId = null;
let activeSentences = [];

function similarity(a, b) {
    const wordsA = normalizeText(a).split(" ");
    const wordsB = normalizeText(b).split(" ");

    const shorter = wordsA.length < wordsB.length ? wordsA : wordsB;
    const longer = wordsA.length < wordsB.length ? wordsB : wordsA;

    let matches = 0;

    shorter.forEach(word => {
        if (word.length > 2 && longer.includes(word)) {
            matches++;
        }
    });

    return matches / Math.max(shorter.length, 1);
}

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
        .replace(/[.,;:!?]/g, "")
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
                normalizedSentence.length > 20 &&
                (
                    normalizedSentence === target ||
                    normalizedSentence.includes(target) ||
                    target.includes(normalizedSentence) ||
                    similarity(normalizedSentence, target) > 0.75
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