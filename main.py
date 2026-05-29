import os
import whisper
import faiss
import numpy as np

from groq import Groq
from sentence_transformers import SentenceTransformer

# =========================================================
# GROQ API CONFIGURATION
# =========================================================

client = Groq(
    api_key="PASTE_YOUR_GROQ_API_KEY_HERE"
)

# =========================================================
# LOAD WHISPER MODEL
# =========================================================

speech_model = whisper.load_model("small")

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# FOLDER PATHS
# =========================================================

AUDIO_FOLDER = "recordings"
TRANSCRIPT_FOLDER = "transcripts"
RESPONSE_FOLDER = "responses"

os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
os.makedirs(RESPONSE_FOLDER, exist_ok=True)

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
# CREATE EMBEDDINGS
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
# RAG RETRIEVAL FUNCTION
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
# GROQ AI RESPONSE FUNCTION
# =========================================================

def generate_ai_response(transcript):

    # ==========================================
    # RETRIEVE RELEVANT KNOWLEDGE
    # ==========================================

    context = retrieve_context(transcript)

    # ==========================================
    # PROMPT
    # ==========================================

    prompt = f"""
You are a professional technical support assistant.

Use the following troubleshooting knowledge
to answer the customer issue accurately.

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
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=700
    )

    return response.choices[0].message.content

# =========================================================
# PROCESS AUDIO FILES
# =========================================================

for audio_file in os.listdir(AUDIO_FOLDER):

    if audio_file.endswith(
        (".wav", ".mp3", ".mpeg", ".m4a")
    ):

        audio_path = os.path.join(
            AUDIO_FOLDER,
            audio_file
        )

        print(f"\n{'=' * 60}")
        print(f"Processing: {audio_file}")
        print(f"{'=' * 60}")

        try:

            # ==========================================
            # SPEECH TO TEXT + TRANSLATION
            # ==========================================

            result = speech_model.transcribe(
                audio_path,
                task="translate"
            )

            transcript = result["text"]

            detected_language = result.get(
                "language",
                "unknown"
            )

            print("\nDetected Language:")
            print(detected_language)

            print("\nTranslated Transcript:")
            print(transcript)

            # ==========================================
            # SAVE TRANSCRIPT
            # ==========================================

            filename = os.path.splitext(
                audio_file
            )[0]

            transcript_path = os.path.join(
                TRANSCRIPT_FOLDER,
                f"{filename}.txt"
            )

            with open(
                transcript_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    f"Detected Language: "
                    f"{detected_language}\n\n"
                )

                file.write(
                    f"Translated Transcript:\n"
                    f"{transcript}"
                )

            # ==========================================
            # GENERATE AI RESPONSE
            # ==========================================

            print(
                "\nGenerating AI Troubleshooting "
                "Response...\n"
            )

            ai_response = generate_ai_response(
                transcript
            )

            print(ai_response)

            # ==========================================
            # SAVE RESPONSE
            # ==========================================

            response_path = os.path.join(
                RESPONSE_FOLDER,
                f"{filename}.txt"
            )

            with open(
                response_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    f"Detected Language: "
                    f"{detected_language}\n\n"
                )

                file.write(
                    f"Translated Transcript:\n"
                    f"{transcript}\n\n"
                )

                file.write(
                    "AI Troubleshooting Response:\n\n"
                )

                file.write(ai_response)

            print(
                f"\nCompleted Successfully: "
                f"{audio_file}"
            )

        except Exception as error:

            print(
                f"\nError processing "
                f"{audio_file}"
            )

            print(error)

print("\nAll audio files processed successfully.")