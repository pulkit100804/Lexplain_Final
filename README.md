# 🏛 Lexplain — Multi-Agent Legal Reasoning Pipeline

Lexplain is a sophisticated legal analysis engine that uses a multi-agent pipeline to ingest case narratives, extract legal facts, retrieve relevant statutes (IPC), and generate structured legal arguments using Gemini 1.5 Flash.

---

## 🚀 Getting Started

Follow these steps to set up and run Lexplain on your local machine.

### 1. Prerequisites
- **Python 3.10** or higher installed.
- A **Google Gemini API Key**. [Get one here](https://aistudio.google.com/app/apikey).

### 2. Setup Environment
Open your terminal in the project root directory and run:

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory and add your Google API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

### 4. Initialize Data (RAG Setup)
Before running the pipeline for the first time, you must index the local judgment database for the Precedent Comparator (Agent 7):

```powershell
python lexplain/cli.py chunk-judgments
```

---

## 🖥 How to Run

### Option A: Web UI (Recommended)
Launch the rich interactive dashboard to visualize agent progress in real-time:

```powershell
python lexplain/app.py
```
- Open your browser to `http://localhost:5000`.
- Enter your case narrative and click **Analyze**.

### Option B: CLI
Run the full pipeline directly from the command line:

```powershell
python lexplain/cli.py run --tenant my_tenant --text "On January 1st, Person A entered Person B's house..."
```

---

## 🤖 Agent Pipeline Architecture
Lexplain operates through a sequenced chain of 14 specialized agents:
1.  **Agent 0 — Ingestion**: Normalizes raw input text.
2.  **Agent 2 — Segmentation**: Breaks text into semantic units.
3.  **Agent 4A/B — Entity & Event Builder**: Extracts actors and actions.
4.  **Agent 5B/C — Statute Retriever**: Finds relevant sections (IPC).
5.  **Agent 6 — Ingredient Evaluator**: Checks fact-statute fit.
6.  **Agent 7 — Precedent Comparator**: Retrieves similar judicial logic.
7.  **Agent 9 — Final Argument Engine**: Generates the final legal brief.
8.  **Agent 5D — Feedback Memory**: Learns from user interactions.

---

## 📂 Project Structure
- `lexplain/app.py`: Flask web application.
- `lexplain/pipeline.py`: Core orchestrator.
- `lexplain/agents/`: Individual agent implementations.
- `lexplain/data/`: Statutes, ingredients, and judgment databases.
- `lexplain/ui/`: Frontend assets.
