# Agentic AI Medical Diagnosis & Triage Assistant

An agentic multi-agent clinical diagnostic assistant grounded in official World Health Organization (WHO) knowledge sheets, utilizing ChromaDB vector search and LLM-powered micro-agents.

---

## 📌 Project Overview

Traditional medical chatbots frequently suffer from hallucinations, lack clinical reasoning steps, and cannot proactively evaluate emergency safety red flags.

This project implements an **Agentic Workflow** that decouples clinical tasks across specialized micro-agents:

- **Clinical Entity Extraction**: Dissects unstructured user complaints into symptoms, duration, and severity.
- **Clarification & Gap Analysis**: Proactively identifies missing clinical variables such as travel history, chronic conditions, and symptom intensity, and prompts for additional details before assessment.
- **Independent Safety Triage**: Acts as an emergency gatekeeper checking for critical red flags such as respiratory failure, altered consciousness, dark urine, and abnormal bleeding before retrieval.
- **Domain-Specific Specialty Routing**: Restricts vector search space to relevant medical specialties to maximize retrieval precision.
- **Strictly Grounded WHO RAG**: Employs ChromaDB to ground clinical insights exclusively in verified WHO fact sheets, explicitly stating when a symptom is unmentioned rather than hallucinating facts.

---

## 🏗️ System Architecture

```text
[ User Query / Symptoms ]
           │
           ▼
┌─────────────────────────┐
│      Symptom Agent      │ ──► Extracts: Entities, Duration, Severity
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Clarification Agent   │ ──► Analyzes missing clinical history & asks questions
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Safety/Triage Agent   │ ──► Safety Gate: Evaluates critical Red Flags & Urgency
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│     Specialty Router    │ ──► Classifies domain (Infectious, Respiratory, Neurology)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│    ChromaDB (WHO RAG)   │ ──► Filtered vector similarity retrieval
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Final Grounded Response │ ──► Structured Output + Testing Advice + Medical Disclaimer
└─────────────────────────┘
Project Structure
AGENTIC MEDICAL DIAGNOSIS/
│
├── medical_knowledge/              # Standardized WHO disease fact sheets
│   ├── infectious_diseases/        # Malaria, Dengue, Influenza, Typhoid, Common Cold
│   ├── neurology/                  # Migraine
│   └── respiratory/                # Asthma, Pneumonia
│
├── rag/
│   └── rag_pipeline.py             # Embeddings, Vector Store, & Retrieval Chain
│
├── agents/
│   ├── __init__.py
│   ├── symptom_agent.py            # Clinical entity extraction
│   ├── clarification_agent.py      # Missing history gap analysis
│   ├── specialty_router.py         # Domain classifier
│   ├── interactive_agent.py        # Terminal orchestrator pipeline
│   └── Safety/
│       ├── __init__.py
│       └── triage_agent.py         # Emergency safety & red flag gatekeeper
│
├── app/
│   └── main.py                     # Streamlit Web User Interface
│
├── vector_db/                      # Persistent ChromaDB vector storage
├── requirements.txt                # Python environment dependencies
└── README.md                       # Project documentation
🔬 Medical Knowledge Base

The vector store is built on standardized clinical data sourced directly from official World Health Organization (WHO) fact sheets:

Specialty	Included Condition	WHO Source Reference
Infectious Diseases	Malaria	WHO Malaria Fact Sheet
	Dengue	WHO Dengue and Severe Dengue Fact Sheet
	Influenza	WHO Seasonal Influenza Fact Sheet
	Typhoid	WHO Typhoid Fact Sheet
	Common Cold	WHO Clinical Guidance on Viral Rhinitis
Neurology	Migraine	WHO Headache Disorders Fact Sheet
Respiratory	Asthma	WHO Asthma Fact Sheet
	Pneumonia	WHO Pneumonia Clinical Guidance
🚀 Installation & Setup
1. Prerequisites
Python 3.10+
Groq API Key
Git
Internet connection for installing Python packages and accessing the LLM API
2. Clone and Setup Environment

Open your terminal in the project directory:

# Optional: Create and activate a virtual environment

python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
3. Environment Variables

Create a .env file in the root directory:

GROQ_API_KEY="your_groq_api_key_here"

Important: Never upload your actual API key to GitHub. Add .env to your .gitignore file.

Example:

.env
venv/
__pycache__/
vector_db/
🖥️ Running the Application
Option A: Streamlit Interactive Web Interface

Run the web application:

python -m streamlit run app/main.py

Open your browser at:

http://localhost:8501

The Streamlit interface provides an interactive way for users to enter symptoms and receive a structured, grounded assessment.

Option B: Terminal-Based CLI Pipeline

Run the multi-agent pipeline directly from the command line:

python -m agents.interactive_agent

This executes the agentic workflow through the terminal.

🤖 Multi-Agent Workflow

The system separates the diagnostic workflow into multiple specialized agents.

1. Symptom Agent

The Symptom Agent processes the user's natural-language complaint and extracts structured information such as:

Symptoms
Duration
Severity
Additional information
Relevant clinical entities

Example:

User Input:
"I have fever, chills and headache for the last three days."

Extracted Information:
Symptoms:
- Fever
- Chills
- Headache

Duration:
- Three days

Severity:
- Not provided
2. Clarification Agent

The Clarification Agent identifies missing information that could be relevant for further assessment.

It may identify gaps such as:

Symptom severity
Travel history
Existing medical conditions
Medication history
Age-related information
Duration or progression of symptoms
Other associated symptoms

The purpose is to reduce uncertainty before generating a final response.

3. Safety/Triage Agent

The Safety/Triage Agent performs an independent safety check before the final assessment.

It looks for potential red flags such as:

Severe difficulty breathing
Altered consciousness
Severe weakness
Abnormal bleeding
Severe dehydration
Other emergency warning signs supported by the knowledge base

If a serious warning sign is detected, the system can prioritize urgent medical evaluation rather than continuing with a routine assessment.

4. Specialty Router

The Specialty Router determines the most relevant medical domain based on the extracted symptoms.

Current supported specialties include:

Infectious Diseases
Respiratory
Neurology
General

Example:

Fever + chills + body pain
        ↓
Infectious Diseases

This classification is then used to filter the ChromaDB retrieval process.

5. WHO-Based RAG Pipeline

The RAG pipeline retrieves relevant information from the project's medical knowledge base.

The process is:

User Symptoms
      ↓
Symptom Extraction
      ↓
Specialty Classification
      ↓
Query Embedding
      ↓
ChromaDB Similarity Search
      ↓
Specialty-Filtered Documents
      ↓
Relevant WHO Knowledge
      ↓
Grounded Response

The system uses vector embeddings to find the most relevant medical knowledge from the stored documents.

🧠 Retrieval-Augmented Generation (RAG)

The project uses Retrieval-Augmented Generation (RAG) to reduce unsupported responses.

Instead of relying only on the language model's internal knowledge, the system:

Receives the user's symptoms.
Converts the query into an embedding.
Searches the ChromaDB vector database.
Retrieves relevant WHO-based knowledge.
Provides the retrieved context to the language model.
Generates a response based on the retrieved information.

This approach helps keep the generated response grounded in the project's knowledge base.

🗄️ ChromaDB Vector Database

ChromaDB is used as the persistent vector database.

Each medical knowledge document is:

Medical Knowledge
       ↓
Document Loading
       ↓
Text Chunking
       ↓
Embedding Generation
       ↓
ChromaDB Storage
       ↓
Similarity Search

The system also stores metadata such as the medical specialty and disease category.

This allows the retrieval process to apply specialty-based filtering.

🎯 Key Design Highlights
1. Deterministic Safety Pre-Checks

Safety checks occur before the final assessment generation so that critical warning signs can trigger urgent escalation advice.

2. Specialty-Filtered Vector Search

ChromaDB metadata filtering limits retrieval to the relevant medical specialty.

This helps reduce irrelevant document retrieval and improves retrieval precision.

3. Anti-Hallucination Constraints

The system is designed to avoid inventing unsupported medical relationships.

When a reported symptom is not supported by the retrieved context, the system can explicitly state that the available source material does not provide evidence for that relationship.

4. Modular Agent Architecture

Each agent has a specific responsibility:

Symptom Agent
      ↓
Clarification Agent
      ↓
Safety/Triage Agent
      ↓
Specialty Router
      ↓
RAG Retrieval
      ↓
Final Response

This modular architecture makes the system easier to test, debug, and extend.

📊 Current Medical Knowledge Coverage

The current knowledge base contains information related to:

Infectious Diseases
Malaria
Dengue
Influenza
Typhoid
Common Cold
Neurology
Migraine
Respiratory Diseases
Asthma
Pneumonia

The knowledge base can be expanded by adding additional verified medical documents and assigning appropriate specialty metadata.

🛠️ Technologies Used
Technology	Purpose
Python	Core programming language
LangChain	RAG and document processing framework
ChromaDB	Vector database
Sentence Transformers	Text embedding generation
Groq	LLM inference
Streamlit	Web interface
python-dotenv	Environment variable management
WHO Fact Sheets	Medical knowledge source
📁 Important Components
rag/rag_pipeline.py

Responsible for:

Loading medical documents
Splitting documents into chunks
Generating embeddings
Creating/updating the ChromaDB vector store
Performing similarity retrieval
Applying specialty-based filtering
agents/symptom_agent.py

Responsible for:

Extracting symptoms
Identifying duration
Identifying severity
Structuring user input
agents/clarification_agent.py

Responsible for:

Detecting missing information
Identifying clinical information gaps
Generating clarification questions
agents/specialty_router.py

Responsible for:

Classifying symptoms into a medical specialty
Routing the query toward the appropriate knowledge domain
agents/Safety/triage_agent.py

Responsible for:

Detecting potential emergency red flags
Performing safety checks
Determining whether urgent escalation advice is required
agents/interactive_agent.py

Responsible for:

Coordinating the different agents
Running the terminal-based workflow
Connecting the individual components into an agentic pipeline
app/main.py

Responsible for:

Streamlit user interface
User symptom input
Displaying the structured system response
🧪 Example Workflow

Example user input:

I have fever, chills, headache and body pain for the last three days.

The system processes the input as:

User Input
    ↓
Symptom Agent
    ↓
Symptoms:
- Fever
- Chills
- Headache
- Body pain
    ↓
Duration:
- Three days
    ↓
Clarification Agent
    ↓
Identifies missing information
    ↓
Safety/Triage Agent
    ↓
Checks for red flags
    ↓
Specialty Router
    ↓
Infectious Diseases
    ↓
ChromaDB
    ↓
WHO-based relevant documents
    ↓
Grounded Final Response
🔐 Security Considerations

The project uses an API key for LLM access.

The following practices should be followed:

Store API keys in .env.
Do not hard-code API keys in Python files.
Do not commit .env to GitHub.
Add .env to .gitignore.
Do not expose API keys in screenshots or project demonstrations.
🔮 Future Scope

The system can be extended with:

Additional WHO medical knowledge documents
More medical specialties
Improved clarification questions
More comprehensive emergency red-flag detection
LangGraph-based agent orchestration
Conversation memory
Patient history tracking
FastAPI backend
Improved Streamlit interface
Structured JSON responses
Additional evaluation datasets
Retrieval accuracy evaluation
Response quality evaluation
Authentication and user management
⚠️ Medical Disclaimer

This project is an academic prototype designed for research and educational purposes only.

It does not provide confirmed medical diagnoses, treatment prescriptions, or formal clinical advice.

The system should not be used as a substitute for a qualified healthcare professional.

Users experiencing serious or emergency symptoms should seek appropriate professional medical care immediately.

👩‍💻 Project Purpose

The primary objective of this project is to demonstrate how Agentic AI, Retrieval-Augmented Generation (RAG), vector databases, LLMs, and specialized AI agents can be combined to build a safety-aware medical information assistant.

The project focuses on:

Agentic AI
     +
Multi-Agent Architecture
     +
RAG
     +
ChromaDB
     +
WHO Knowledge
     +
LLM
     +
Safety Triage
     =
Grounded Medical Information Assistant
⭐ Conclusion

The Agentic AI Medical Diagnosis & Triage Assistant demonstrates a modular approach to building an AI-based medical information system.

By combining specialized agents, WHO-grounded retrieval, ChromaDB vector search, specialty routing, and safety checks, the project aims to provide structured and source-grounded responses while reducing unsupported medical claims.

The architecture is designed to be modular and extensible, allowing additional diseases, agents, medical knowledge sources, evaluation methods, and user-interface capabilities to be incorporated in future versions.