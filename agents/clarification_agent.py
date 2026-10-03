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
# CLARIFICATION PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a Clarification Agent in a medical information
system.

Your task is to examine the symptoms provided by the user
and identify important missing information.

Do not provide a diagnosis.

Do not invent information.

Ask only relevant questions that can help understand the
user's symptoms better.

Focus on:

1. Duration of symptoms
2. Severity
3. Important associated symptoms
4. Relevant information that is missing

If the information is already provided, do not ask for it again.

User Symptoms:
{symptoms}

Provide your response in this format:

Missing Information:
- ...

Clarification Questions:
1. ...
2. ...
3. ...

If no additional information is required, write:
"No major clarification required."
"""
)


# --------------------------------------------------
# CREATE AGENT
# --------------------------------------------------

clarification_agent = (
    prompt
    | llm
    | StrOutputParser()
)


# --------------------------------------------------
# FUNCTION
# --------------------------------------------------

def get_clarification_questions(symptoms):

    result = clarification_agent.invoke({
        "symptoms": symptoms
    })

    return result


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    symptoms = """
    Fever and headache.
    """

    print("=" * 60)
    print("CLARIFICATION AGENT")
    print("=" * 60)

    result = get_clarification_questions(symptoms)

    print("\n")
    print(result)

    print("\n" + "=" * 60)
    print("CLARIFICATION AGENT TEST COMPLETED")
    print("=" * 60)