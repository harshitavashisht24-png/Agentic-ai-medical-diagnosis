import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# GROQ API KEY
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        f"GROQ_API_KEY not found. Please check your .env file at: {ENV_FILE}"
    )


# ============================================================
# PATHS
# ============================================================

KNOWLEDGE_PATH = PROJECT_ROOT / "medical_knowledge"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("Starting Medical RAG Pipeline...")
print("=" * 60)


# ============================================================
# LOAD MEDICAL DOCUMENTS
# ============================================================

documents = []

for file_path in KNOWLEDGE_PATH.rglob("*.txt"):

    # Ignore non-medical helper files
    if file_path.name.lower() in [
        "readme.txt",
        "specialty_mapping.txt"
    ]:
        continue

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    file_documents = loader.load()

    for doc in file_documents:

        # Determine specialty from folder structure
        relative_path = file_path.relative_to(KNOWLEDGE_PATH)

        if len(relative_path.parts) > 1:
            specialty = relative_path.parts[0]
        else:
            specialty = "general"

        # Disease name from filename
        disease = file_path.stem

        # Add metadata
        doc.metadata["specialty"] = specialty
        doc.metadata["disease"] = disease

        documents.append(doc)


print(f"\nDocuments loaded: {len(documents)}")


# ============================================================
# TEXT CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)


print("\nMedical Knowledge Files:")

for doc in documents:

    print(
        f"Disease: {doc.metadata.get('disease')} | "
        f"Specialty: {doc.metadata.get('specialty')}"
    )


print(f"\nDocuments loaded: {len(documents)}")
print(f"Total chunks created: {len(chunks)}")


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("Embedding model ready!")


# ============================================================
# CHROMADB VECTOR STORE
# ============================================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(VECTOR_DB_PATH)
)


print("ChromaDB vector store ready!")


# ============================================================
# SPECIALTY RETRIEVER
# ============================================================

def get_specialty_retriever(specialty):
    """
    Creates a retriever that searches only within
    the selected medical specialty.
    """

    # General search
    if specialty == "general":

        return vectorstore.as_retriever(
            search_kwargs={
                "k": 3
            }
        )

    # Specialty-filtered search
    return vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {
                "specialty": specialty
            }
        }
    )


print("Vector store retriever ready!")


# ============================================================
# GROQ LLM
# ============================================================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0
)


print("Groq LLM ready!")


# ============================================================
# SPECIALTY ROUTER
# ============================================================

router_prompt = ChatPromptTemplate.from_template(
    """
You are a medical specialty routing assistant.

Your task is ONLY to identify which medical specialty
is most relevant to the user's symptoms.

Available specialties:

- infectious_diseases
- respiratory
- neurology
- general

Rules:

1. Return ONLY ONE specialty name.
2. Do not provide a diagnosis.
3. Do not provide medical advice.
4. Do not explain your answer.
5. If the symptoms do not clearly match a specialty,
   return general.

Examples:

Fever, chills, sweating, body pain
-> infectious_diseases

Cough, breathing difficulty, wheezing
-> respiratory

Severe headache, sensitivity to light
-> neurology

Unclear or unrelated symptoms
-> general

User symptoms:

{symptoms}

Specialty:
"""
)


router_chain = (
    router_prompt
    | llm
    | StrOutputParser()
)


def route_specialty(symptoms):
    """
    Automatically selects the most relevant
    medical specialty based on symptoms.
    """

    specialty = router_chain.invoke(
        {
            "symptoms": symptoms
        }
    )

    specialty = specialty.strip().lower()

    allowed_specialties = [
        "infectious_diseases",
        "respiratory",
        "neurology",
        "general"
    ]

    for allowed in allowed_specialties:

        if allowed in specialty:
            return allowed

    return "general"


# ============================================================
# MEDICAL RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an AI medical information assistant for an educational
medical RAG system.

Your task is to analyze the user's reported symptoms using ONLY
the medical information provided in the retrieved context.

IMPORTANT RULES:

1. Do NOT provide or claim a confirmed medical diagnosis.

2. Identify only possible conditions that are supported by the
   retrieved medical context.

3. Do NOT introduce any disease, symptom, warning sign, test,
   treatment, medicine, or medical fact that is not present in
   the retrieved context.

4. Clearly distinguish between:
   - symptoms that overlap with a condition, and
   - a confirmed diagnosis.
   Symptom overlap alone must never be presented as confirmation.

5. Explain how the user's reported symptoms relate to the
   retrieved medical information.

6. Mention relevant warning signs ONLY when they are present
   in the retrieved context.

7. Do NOT state that the user does not have warning signs unless
   the user has explicitly provided enough information to make
   that statement.

8. If the retrieved information is insufficient to determine
   whether a warning sign is present or absent, say:
   "The available information is insufficient to assess this."

9. If the retrieved information is insufficient, clearly state
   that additional information, clinical evaluation, or medical
   testing may be required.

10. Do not invent medical facts or make assumptions about the
    user's age, gender, medical history, medications, pregnancy
    status, or other personal circumstances.

11. Do not recommend a specific medicine, dosage, or treatment
    unless it is explicitly supported by the retrieved context.

12. Recommend consultation with a qualified healthcare
    professional when appropriate, especially when symptoms are
    severe, persistent, worsening, or associated with warning
    signs mentioned in the retrieved context.

13. Keep the response concise, clear, and easy to understand.

14. This system provides educational information only.

Retrieved Medical Context:

{context}

User Symptoms:

{question}

Provide the response using exactly this structure:

Possible Conditions:
- List only conditions supported by the retrieved context.
- Explain that these are possibilities based on symptom overlap,
  not confirmed diagnoses.

Symptom Analysis:
- Explain which reported symptoms are supported by the retrieved
  context.
- Mention important differences or missing information when
  relevant.

Warning Signs:
- Mention warning signs from the retrieved context that the user
  has actually reported.
- If none have been reported, say:
  "No warning signs were reported in the provided symptoms."
- If the information is insufficient to determine whether a warning
  sign is present, say:
  "The available information is insufficient to assess this."

Recommendation:
- Provide only recommendations supported by the retrieved context.
- Mention when medical evaluation or testing may be appropriate.

Medical Disclaimer:
This system provides educational information only and does not
provide a confirmed medical diagnosis or individualized treatment.
"""
)

# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# RAG CHAIN
# ============================================================

def create_rag_chain(specialty="general"):

    specialty_retriever = get_specialty_retriever(
        specialty
    )

    def retrieve_and_format(inputs):

        question = inputs["question"]

        retrieved_docs = specialty_retriever.invoke(
            question
        )

        context = format_docs(
            retrieved_docs
        )

        return {
            "context": context,
            "question": question
        }

    rag_chain = (
        retrieve_and_format
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
def remove_duplicate_diseases(documents):
    """
    Keeps only the first retrieved chunk for each disease.
    This prevents the same disease from appearing multiple times.
    """

    unique_documents = []
    seen_diseases = set()

    for doc in documents:

        disease = doc.metadata.get(
            "disease",
            "unknown"
        )

        if disease not in seen_diseases:

            seen_diseases.add(
                disease
            )

            unique_documents.append(
                doc
            )

    return unique_documents

# ============================================================
# MEDICAL RAG FUNCTION
# ============================================================

def medical_rag(
    symptoms,
    specialty="general"
):
    """
    Performs specialty-based medical retrieval
    and generates an educational assessment.
    """

    # Create specialty-specific retriever
    specialty_retriever = get_specialty_retriever(
        specialty
    )

    # Retrieve relevant documents
    retrieved_docs = specialty_retriever.invoke(
        symptoms
    )

    # Remove duplicate diseases
    retrieved_docs = remove_duplicate_diseases(
        retrieved_docs
    )

    # Convert documents into context
    context = format_docs(
        retrieved_docs
    )

    # Generate response using RAG prompt + LLM
    response = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke(
        {
            "context": context,
            "question": symptoms
        }
    )

    return response

# ============================================================
# GET RETRIEVED DOCUMENTS
# ============================================================

def get_retrieved_documents(
    symptoms,
    specialty="general"
):

    specialty_retriever = get_specialty_retriever(
        specialty
    )

    retrieved_docs = specialty_retriever.invoke(
        symptoms
    )

    # Remove duplicate diseases
    retrieved_docs = remove_duplicate_diseases(
        retrieved_docs
    )

    return retrieved_docs


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # TEST SYMPTOMS
    # --------------------------------------------------------

    query = (
        "I have fever, chills, sweating, headache, "
        "weakness, fatigue and muscle aches."
    )


    # --------------------------------------------------------
    # USER QUERY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("USER QUERY")
    print("=" * 60)

    print(query)


    # --------------------------------------------------------
    # SPECIALTY ROUTER
    # --------------------------------------------------------

    specialty = route_specialty(
        query
    )


    print("\n" + "=" * 60)
    print("SPECIALTY ROUTER")
    print("=" * 60)

    print(
        f"Selected Specialty: {specialty}"
    )


    # --------------------------------------------------------
    # RETRIEVED DOCUMENTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RETRIEVED MEDICAL DOCUMENTS")
    print("=" * 60)


    retrieved_docs = get_retrieved_documents(
        query,
        specialty
    )


    for i, doc in enumerate(
        retrieved_docs
    ):

        print(
            f"\n[{i + 1}]"
        )

        print(
            f"Disease: "
            f"{doc.metadata.get('disease')}"
        )

        print(
            f"Specialty: "
            f"{doc.metadata.get('specialty')}"
        )

        print(
            doc.page_content[:500]
        )

        print(
            "-" * 50
        )


    # --------------------------------------------------------
    # GENERATED MEDICAL ASSESSMENT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GENERATED MEDICAL ASSESSMENT")
    print("=" * 60)


    response = medical_rag(
        query,
        specialty
    )


    print(response)


    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETED")
    print("=" * 60)