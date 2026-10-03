import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


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
# LLM
# ============================================================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# SPECIALTY ROUTING PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a medical specialty routing agent for an educational
Agentic AI prototype.

Your task is to select the most relevant medical specialty
knowledge domain based on the user's symptom information.

Available specialties:

- cardiology
- oncology
- neurology
- orthopedics
- respiratory
- infectious_diseases
- general

- Use infectious_diseases for symptoms or conditions associated with infectious diseases such as dengue, malaria, influenza, typhoid, or common cold.

IMPORTANT RULES:

1. Do NOT diagnose a disease.
2. Only route the information to the most relevant specialty.
3. Use only the symptoms and information provided by the user.
4. If the information is insufficient or ambiguous, select "general".
5. Do not invent symptoms or medical history.
6. Return ONLY valid JSON.
7. The JSON must follow exactly this structure:

{{
    "specialty": "specialty_name",
    "reason": "short explanation"
}}

User symptom information:

{input}
"""
)


# ============================================================
# SPECIALTY ROUTER FUNCTION
# ============================================================

def route_to_specialty(user_input):

    messages = prompt.format_messages(
        input=user_input
    )

    response = llm.invoke(messages)

    raw_response = response.content.strip()

    try:
        result = json.loads(raw_response)

        allowed_specialties = {
            "cardiology",
            "oncology",
            "neurology",
            "orthopedics",
            "respiratory",
            "infectious_diseases",
            "general"
        }

        specialty = result.get("specialty", "general").lower()

        if specialty not in allowed_specialties:
            specialty = "general"

        return {
            "specialty": specialty,
            "reason": result.get(
                "reason",
                "Insufficient information for specialty routing."
            )
        }

    except json.JSONDecodeError:

        return {
            "specialty": "general",
            "reason": "The system could not confidently determine a specialty."
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SPECIALTY ROUTER AGENT")
    print("=" * 60)

    user_input = input("\nEnter symptoms: ")

    result = route_to_specialty(user_input)

    print("\nSPECIALTY ROUTING RESULT")
    print("-" * 60)

    print(json.dumps(result, indent=4))

    print("\n" + "=" * 60)
    print("SPECIALTY ROUTER TEST COMPLETED")
    print("=" * 60)