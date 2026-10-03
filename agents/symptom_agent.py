import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
# SYMPTOM ANALYSIS PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a Symptom Analysis Agent in a medical information
system.

Your task is to analyze the symptoms provided by the user.

Extract and organize the information into:

1. Symptoms
2. Duration
3. Severity
4. Other relevant information

Do not provide a medical diagnosis.

Do not invent symptoms or information that the user did
not provide.

If duration or severity is not provided, write:
"Not provided"

User Input:
{user_input}

Return the result in this format:

Symptoms:
- ...

Duration:
- ...

Severity:
- ...

Other Information:
- ...
"""
)


# --------------------------------------------------
# CREATE AGENT
# --------------------------------------------------

symptom_agent = (
    prompt
    | llm
    | StrOutputParser()
)


# --------------------------------------------------
# FUNCTION
# --------------------------------------------------

def analyze_symptoms(user_input):

    result = symptom_agent.invoke({
        "user_input": user_input
    })

    return result


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    user_input = (
        "I have fever, chills, headache and body pain "
        "for the last three days."
    )

    print("=" * 60)
    print("SYMPTOM ANALYSIS AGENT")
    print("=" * 60)

    result = analyze_symptoms(user_input)

    print("\n")
    print(result)

    print("\n" + "=" * 60)
    print("AGENT TEST COMPLETED")
    print("=" * 60)