import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# --------------------------------------------------
# IMPORT AGENTS
# --------------------------------------------------

from agents.symptom_agent import analyze_symptoms
from agents.clarification_agent import get_clarification_questions
from agents.Safety.triage_agent import assess_safety

from rag.rag_pipeline import medical_rag


# --------------------------------------------------
# MAIN MEDICAL AGENT WORKFLOW
# --------------------------------------------------

def medical_agent(user_input):

    print("\n" + "=" * 60)
    print("STEP 1 - SYMPTOM ANALYSIS")
    print("=" * 60)

    symptom_result = analyze_symptoms(user_input)

    print(symptom_result)


    # --------------------------------------------------
    # STEP 2 - CLARIFICATION
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 2 - CLARIFICATION ANALYSIS")
    print("=" * 60)

    clarification_result = get_clarification_questions(
        symptom_result
    )

    print(clarification_result)


    # --------------------------------------------------
    # STEP 3 - SAFETY / TRIAGE
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 3 - SAFETY / TRIAGE")
    print("=" * 60)

    triage_result = assess_safety(
        symptom_result
    )

    print(triage_result)


    # --------------------------------------------------
    # STEP 4 - RAG MEDICAL KNOWLEDGE
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 4 - MEDICAL KNOWLEDGE RETRIEVAL")
    print("=" * 60)

    rag_result = medical_rag(
        user_input
    )

    print(rag_result)


    # --------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL MEDICAL INFORMATION")
    print("=" * 60)

    print(rag_result)

    return rag_result


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    user_input = (
        "I have fever, chills, headache, weakness and "
        "body pain for the last three days."
    )

    medical_agent(user_input)