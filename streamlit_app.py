import os
import tempfile
from pathlib import Path

import streamlit as st
import whisper
import faiss
import numpy as np

from groq import Groq
from sentence_transformers import SentenceTransformer

# =========================================================
# FOLDER CONFIGURATION
# =========================================================

AUDIO_FOLDER = Path("recordings")
TRANSCRIPT_FOLDER = Path("transcripts")
RESPONSE_FOLDER = Path("responses")

AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_FOLDER.mkdir(parents=True, exist_ok=True)
RESPONSE_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================================================
# SUPPORTED AUDIO TYPES
# =========================================================

ALLOWED_AUDIO_TYPES = ("wav", "mp3", "mpeg", "m4a")

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

with open(
    "knowledge_base.txt",
    "r",
    encoding="utf-8"
) as file:

    knowledge_chunks = file.readlines()

# =========================================================
# CREATE KNOWLEDGE EMBEDDINGS
# =========================================================

knowledge_embeddings = embedding_model.encode(
    knowledge_chunks
)

# =========================================================
# CREATE FAISS VECTOR DATABASE
# =========================================================

dimension = knowledge_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(knowledge_embeddings)
)

# =========================================================
# LOAD WHISPER MODEL
# =========================================================

@st.cache_resource
def load_speech_model():

    return whisper.load_model("small")

# =========================================================
# CREATE GROQ CLIENT
# =========================================================

def get_groq_client(api_key):

    if not api_key:
        return None

    return Groq(api_key=api_key)

# =========================================================
# RETRIEVE CONTEXT USING RAG
# =========================================================

def retrieve_context(query):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        k=2
    )

    retrieved_context = []

    for idx in indices[0]:

        retrieved_context.append(
            knowledge_chunks[idx]
        )

    return "\n".join(retrieved_context)

# =========================================================
# TRANSCRIBE AUDIO
# =========================================================

def transcribe_audio(file_path: Path):

    speech_model = load_speech_model()

    result = speech_model.transcribe(
        str(file_path),
        task="translate"
    )

    transcript = result.get("text", "")

    detected_language = result.get(
        "language",
        "unknown"
    )

    return transcript, detected_language

# =========================================================
# GENERATE AI RESPONSE
# =========================================================

def generate_ai_response(
    transcript: str,
    client: Groq
):

    # ==========================================
    # RETRIEVE KNOWLEDGE
    # ==========================================

    context = retrieve_context(transcript)

    # ==========================================
    # PROMPT
    # ==========================================

    prompt = f"""
You are a professional technical support assistant.

Use the retrieved troubleshooting knowledge
to answer the issue accurately.

Retrieved Knowledge:
{context}

Customer Issue:
{transcript}

Provide ONLY in this format:

Problem Summary:
[summary]

Possible Causes:
- Cause 1
- Cause 2
- Cause 3

Step-by-Step Troubleshooting:
1. Step one
2. Step two
3. Step three

Final Recommendation:
[final recommendation]

Do not explain translation.
Do not mention languages.
Do not add extra commentary.
"""

    # ==========================================
    # GROQ LLM CALL
    # ==========================================

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.3,
        max_tokens=700,
    )

    return response.choices[0].message.content

# =========================================================
# SAVE TEXT FILE
# =========================================================

def save_text_file(
    folder: Path,
    filename: str,
    content: str
):

    path = folder / filename

    path.write_text(
        content,
        encoding="utf-8"
    )

    return path

# =========================================================
# LIST AUDIO FILES
# =========================================================

def list_audio_files():

    return sorted(
        [
            file
            for file in AUDIO_FOLDER.iterdir()
            if file.suffix.lstrip(".").lower()
            in ALLOWED_AUDIO_TYPES
        ]
    )

# =========================================================
# LIST TEXT FILES
# =========================================================

def list_text_files(folder: Path):

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.suffix == ".txt"
        ]
    )

# =========================================================
# SIDEBAR FILE VIEWER
# =========================================================

def render_existing_files():

    st.sidebar.header("Existing Files")

    with st.sidebar.expander(
        "Transcripts",
        expanded=False
    ):

        for transcript_file in list_text_files(
            TRANSCRIPT_FOLDER
        ):

            if st.button(
                f"View {transcript_file.name}",
                key=f"trans_{transcript_file.name}"
            ):

                st.session_state[
                    "open_file"
                ] = transcript_file

    with st.sidebar.expander(
        "Responses",
        expanded=False
    ):

        for response_file in list_text_files(
            RESPONSE_FOLDER
        ):

            if st.button(
                f"View {response_file.name}",
                key=f"resp_{response_file.name}"
            ):

                st.session_state[
                    "open_file"
                ] = response_file

    open_file = st.session_state.get(
        "open_file"
    )

    if open_file:

        st.markdown(
            f"### {open_file.name}"
        )

        st.code(
            open_file.read_text(
                encoding="utf-8"
            ),
            language="text"
        )

# =========================================================
# PROCESS AUDIO FILE
# =========================================================

def process_audio_file(
    audio_path: Path,
    client: Groq
):

    # ==========================================
    # TRANSCRIBE AUDIO
    # ==========================================

    transcript, detected_language = (
        transcribe_audio(audio_path)
    )

    # ==========================================
    # SAVE TRANSCRIPT
    # ==========================================

    transcript_text = (
        f"Detected Language: "
        f"{detected_language}\n\n"
        f"Translated Transcript:\n"
        f"{transcript}\n"
    )

    transcript_path = save_text_file(
        TRANSCRIPT_FOLDER,
        audio_path.with_suffix(".txt").name,
        transcript_text
    )

    # ==========================================
    # GENERATE AI RESPONSE
    # ==========================================

    ai_response_text = generate_ai_response(
        transcript,
        client
    )

    # ==========================================
    # SAVE RESPONSE
    # ==========================================

    response_text = (
        f"Detected Language: "
        f"{detected_language}\n\n"
        f"Translated Transcript:\n"
        f"{transcript}\n\n"
        f"AI Troubleshooting Response:\n\n"
        f"{ai_response_text}"
    )

    response_path = save_text_file(
        RESPONSE_FOLDER,
        audio_path.with_suffix(".txt").name,
        response_text
    )

    return (
        transcript,
        detected_language,
        ai_response_text,
        transcript_path,
        response_path,
    )

# =========================================================
# MAIN STREAMLIT APP
# =========================================================

def main():

    st.set_page_config(
        page_title="AI Remote Troubleshooting Assistant",
        layout="wide"
    )

    st.title(
        "AI Remote Troubleshooting Assistant"
    )

    st.markdown(
        """
This application uses:
- Whisper AI for speech recognition
- RAG pipeline using FAISS
- Groq Llama 3 for AI troubleshooting
"""
    )

    # ==========================================
    # API KEY SECTION
    # ==========================================

    env_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    st.sidebar.header("Groq API Key")

    if env_api_key:

        st.sidebar.success(
            "Detected GROQ_API_KEY "
            "from environment."
        )

    else:

        st.sidebar.info(
            "Paste your Groq API key below."
        )

    sidebar_key = st.sidebar.text_input(
        "Groq API key",
        type="password",
        placeholder="Paste API key here",
    )

    api_key = env_api_key or sidebar_key.strip()

    client = get_groq_client(api_key)

    if client is None:

        st.warning(
            "Groq API key missing."
        )

    # ==========================================
    # AUDIO SELECTION
    # ==========================================

    st.sidebar.header("Audio Selection")

    available_files = list_audio_files()

    audio_names = [
        audio.name
        for audio in available_files
    ]

    audio_choice = st.sidebar.selectbox(
        "Choose existing recording",
        ["-- Select --"] + audio_names
    )

    upload_file = st.sidebar.file_uploader(
        "Or upload audio",
        type=list(ALLOWED_AUDIO_TYPES)
    )

    chosen_audio_path = None

    uploaded_local_path = None

    if upload_file is not None:

        suffix = (
            Path(upload_file.name).suffix
            or ".wav"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp_file:

            tmp_file.write(upload_file.read())

            uploaded_local_path = Path(
                tmp_file.name
            )

        chosen_audio_path = uploaded_local_path

        st.success(
            f"Uploaded file: "
            f"{upload_file.name}"
        )

    elif audio_choice != "-- Select --":

        chosen_audio_path = (
            AUDIO_FOLDER / audio_choice
        )

    # ==========================================
    # PROCESS AUDIO
    # ==========================================

    if chosen_audio_path:

        st.subheader("Selected Audio")

        st.audio(str(chosen_audio_path))

        if st.button(
            "Process Audio",
            key="process_audio"
        ):

            if client is None:

                st.error(
                    "Groq API key required."
                )

            else:

                with st.spinner(
                    "Processing audio..."
                ):

                    (
                        transcript,
                        detected_language,
                        ai_response,
                        transcript_path,
                        response_path,
                    ) = process_audio_file(
                        chosen_audio_path,
                        client
                    )

                st.success(
                    "Processing Complete!"
                )

                # ==============================
                # TRANSCRIPT
                # ==============================

                st.markdown(
                    "## Transcript"
                )

                st.write(
                    f"Detected Language: "
                    f"{detected_language}"
                )

                st.text_area(
                    "Translated Transcript",
                    transcript,
                    height=200
                )

                # ==============================
                # AI RESPONSE
                # ==============================

                st.markdown(
                    "## AI Troubleshooting Response"
                )

                st.text_area(
                    "Troubleshooting Response",
                    ai_response,
                    height=350
                )

                # ==============================
                # SAVED FILES
                # ==============================

                st.markdown("---")

                st.write(
                    f"Saved transcript: "
                    f"`{transcript_path}`"
                )

                st.write(
                    f"Saved response: "
                    f"`{response_path}`"
                )

    # ==========================================
    # EXISTING FILES
    # ==========================================

    st.sidebar.markdown("---")

    render_existing_files()

    # ==========================================
    # FOOTER
    # ==========================================

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Whisper + RAG + Groq Llama 3"
    )

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()