import pysbd


def chunk_with_pysbd(long_english_text: str, max_chars: int = 3000) -> list[str]:
    """
    Uses pySBD to split long text into sentences, then groups them into
    chunks that respect the 2K context limit (approximated by max_chars).
    """
    # 1. Initialize the pySBD segmenter for English
    segmenter = pysbd.Segmenter(language="en", clean=False)

    # 2. Extract accurate sentence boundaries
    sentences = segmenter.segment(long_english_text)

    chunks = []
    current_chunk = ""

    # 3. Aggregate sentences into safe-sized chunks
    for sentence in sentences:
        # If a single sentence is bizarrely long, we must force it in anyway
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Catch the final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# Example Integration into the Inference Pipeline
def chunk_and_translate(long_english_text: str, model, processor) -> str:
    # Use max_chars=3000 to safely stay under Gemma's ~2K token limit
    chunks = chunk_with_pysbd(long_english_text, max_chars=3000)

    translated_chunks = []
    for chunk in chunks:
        translated = run_inference(chunk, model, processor)
        translated_chunks.append(translated)

    return " ".join(translated_chunks)
