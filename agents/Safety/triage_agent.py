import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found.")


# --------------------------------------------------
# GROQ MODEL
# --------------------------------------------------

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# TRIAGE PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a Safety and Triage Agent in a medical
information system.

Your task is to identify whether the user's reported
symptoms contain potentially concerning warning signs.

IMPORTANT:

1. Do not provide a medical diagnosis.
2. Do not claim that the user has a disease.
3. Do not invent symptoms.
4. Do not provide treatment instructions.
5. If potentially serious symptoms are present, clearly
   recommend appropriate medical evaluation.
6. If no concerning symptoms are mentioned, say so clearly.
7. This is an educational system and not a replacement
   for a healthcare professional.

Consider symptoms such as:

- Difficulty breathing
- Severe chest pain
- Loss of consciousness
- Severe confusion
- Severe or sudden weakness
- Seizures
- Severe dehydration
- Very severe or rapidly worsening symptoms

User Symptoms:
{symptoms}

Return the response in this format:

Triage Level:
- LOW / MODERATE / HIGH

Warning Signs:
- ...

Reason:
- ...

Recommended Action:
- ...

Medical Disclaimer:
- This system provides educational information only and
  does not provide a confirmed diagnosis or treatment.
"""
)


# --------------------------------------------------
# CREATE TRIAGE AGENT
# --------------------------------------------------

triage_agent = (
    prompt
    | llm
    | StrOutputParser()
)


# --------------------------------------------------
# FUNCTION
# --------------------------------------------------

def assess_safety(symptoms):

    result = triage_agent.invoke({
        "symptoms": symptoms
    })

    return result


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    symptoms = """
    I have fever, headache and body pain for three days.
    """

    print("=" * 60)
    print("SAFETY / TRIAGE AGENT")
    print("=" * 60)

    result = assess_safety(symptoms)

    print("\n")
    print(result)

    print("\n" + "=" * 60)
    print("TRIAGE AGENT TEST COMPLETED")
    print("=" * 60)