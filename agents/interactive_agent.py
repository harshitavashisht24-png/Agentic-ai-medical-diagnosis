from agents.symptom_agent import analyze_symptoms
from agents.clarification_agent import get_clarification_questions
from agents.Safety.triage_agent import assess_safety
from rag.rag_pipeline import medical_rag


def interactive_medical_agent():

    print("\n" + "=" * 60)
    print("AGENTIC AI FOR MEDICAL INFORMATION")
    print("=" * 60)

    # STEP 1: USER INPUT
    user_input = input(
        "\nEnter your symptoms:\n> "
    )

    if not user_input.strip():
        print("No symptoms were entered.")
        return

    # STEP 2: SYMPTOM ANALYSIS
    print("\n" + "=" * 60)
    print("STEP 1 - SYMPTOM ANALYSIS")
    print("=" * 60)

    symptom_result = analyze_symptoms(user_input)

    print(symptom_result)

    # STEP 3: CLARIFICATION
    print("\n" + "=" * 60)
    print("STEP 2 - CLARIFICATION")
    print("=" * 60)

    clarification_result = get_clarification_questions(
        symptom_result
    )

    print(clarification_result)

    # STEP 4: USER ANSWERS
    clarification_answers = input(
        "\nPlease provide answers to the above questions "
        "(or type 'skip'):\n> "
    )

    if clarification_answers.strip().lower() == "skip":
        clarification_answers = "No additional information provided."

    # STEP 5: COMBINE INFORMATION
    combined_information = f"""
Original User Symptoms:
{user_input}

Symptom Analysis:
{symptom_result}

Clarification Answers:
{clarification_answers}
"""

    # STEP 6: SAFETY / TRIAGE
    print("\n" + "=" * 60)
    print("STEP 3 - SAFETY / TRIAGE")
    print("=" * 60)

    triage_result = assess_safety(
        combined_information
    )

    print(triage_result)

    # STEP 7: RAG
    print("\n" + "=" * 60)
    print("STEP 4 - MEDICAL KNOWLEDGE RETRIEVAL")
    print("=" * 60)

    rag_result = medical_rag(
        combined_information
    )

    print(rag_result)

    # FINAL OUTPUT
    print("\n" + "=" * 60)
    print("FINAL MEDICAL INFORMATION")
    print("=" * 60)

    print(rag_result)

    print("\nIMPORTANT:")
    print(
        "This system is an educational prototype and does not "
        "provide a confirmed medical diagnosis or treatment."
    )


if __name__ == "__main__":
    interactive_medical_agent()