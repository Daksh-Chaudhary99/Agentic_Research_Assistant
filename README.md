---
title: Agentic Research Assistant
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
python_version: 3.11
app_file: app.py
pinned: false
tags:
  - agent-demo-track
  - Mistral
  - LlamaIndex
  - Multi-agent
short_description: A Multi-Agent based Research Assistant
---

# 🤖 AI Research Assistant

This project is a powerful, AI-powered web application designed to accelerate the process of academic research. It acts as an intelligent partner for researchers, graduate students, and anyone looking to quickly understand and explore scientific literature. The application is built using a modern agentic framework with LlamaIndex and is powered by the Mistral Large Language Model.

Link to Project Demo: https://www.dropbox.com/scl/fi/f5dcler2ormu0h2tt4vgj/DakshC-Agentic-Research-Assistant-Made-with-Clipchamp.mp4?rlkey=jlk29ylgv5c8c3l4syzbz5ic5&st=1llckwas&dl=0

---

## ✨ Features

This Research Assistant offers two primary workflows accessible through a clean, tab-based interface:

### 1. Analyze a Specific PDF
- **Deep-Dive Analysis:** Users can upload a research paper directly in PDF format.
- **Structured Reports:** The application performs a comprehensive analysis and generates a detailed, multi-section technical brief covering the paper's core ideas, methodology, results, and potential future research directions.
- **Citation Tool:** After an analysis is complete, a tool appears allowing the user to instantly generate and copy a correctly formatted **BibTeX citation** for the paper, ready to be used in reference managers like Zotero or in LaTeX documents.

### 2. Explore a Research Topic
- **Automated Literature Discovery:** Users can enter a research topic or a simple query (e.g., "AI Agents for Software Engineering").
- **Agentic Web Search:** A specialized **Scout Agent** uses the Tavily web search tool to find recent, relevant papers from academic sources like `arxiv.org`.
- **Intelligent Summarization:** The Scout Agent doesn't just return a list of links; it synthesizes the search results and provides a high-level summary of the key papers and findings, complete with inline citations, helping the user quickly grasp the state of the field.

---

## ⚙️ How It Works: The Technology Stack

This application is built on a modern, agentic architecture designed for complex reasoning and tool use.

### Core Frameworks
- **LlamaIndex:** Serves as the central data framework to connect the language models with our data (PDFs) and tools (web search). It provides the `VectorStoreIndex`, `QueryEngine`, and `ReActAgent` components that are the building blocks of the application.
- **Gradio:** Used to create the interactive and user-friendly web interface.
- **Mistral AI:** The entire application's intelligence is powered by the **Mistral LLM**. It is used for all reasoning, synthesis, summarization, and data extraction tasks.
  - **Language Model (`mistral-medium-latest`):** The "brain" of the agents and the analysis engine.
  - **Embedding Model (`mistral-embed`):** Used during the analysis of a PDF to convert the document's text into numerical vectors. This allows the system to understand the semantic meaning of the text and find the most relevant context to answer complex questions.

### The AI Agents and Their Tasks

The application employs a sophisticated system of specialized AI agents, each with a specific role and prompt.

#### 1. The Scout Agent (`Explore a Research Topic`)
- **Task:** To act as a research scout. When given a topic, its job is to find relevant papers and provide an initial summary.
- **Tools:** It is equipped with the **Tavily Search API**, which it uses to perform targeted web searches, focusing on `arxiv.org`.
- **Functionality:** The agent receives the user's query, uses its search tool to gather information, and then uses the Mistral LLM to reason over the search results and synthesize the final summary, as guided by its system prompt (`SCOUT_PROMPT` in `agents.py`).

#### 2. The Analysis Workflow (`Analyze a Specific PDF`)
- **Task:** To perform a deep, multi-faceted analysis of a single research paper.
- **Functionality:** This workflow, defined in `analysis.py`, uses a stable and efficient single-query engine approach:
    1.  **Indexing:** The uploaded PDF is parsed and converted into a `VectorStoreIndex` using Mistral's embedding model. This creates a searchable knowledge base of the paper's content.
    2.  **Querying:** A LlamaIndex `QueryEngine` is created. It is given a single, comprehensive prompt (`COMPREHENSIVE_ANALYSIS_PROMPT`) that instructs it to generate a full, multi-section report covering everything from the core problem to future work. This single-prompt approach is robust and performant.

#### 3. The Citation Extractor (`Export Citation` Tool)
- **Task:** A highly specialized, tool-less task to extract specific bibliographic data.
- **Functionality:** When the user clicks "Export Citation," this workflow makes a direct call to the Mistral LLM. It uses the `CITATION_EXTRACTOR_PROMPT`, which strictly instructs the model to find the title, authors, and year from the first page of the paper and return it **only** as a JSON object. This structured data is then passed to a deterministic Python function (`format_to_bibtex` in `utils.py`) to generate the final citation.

---

## 🚀 How to Use

1.  **To Analyze a Paper:**
    - Select the **"Analyze a Specific PDF"** tab.
    - Upload your PDF file.
    - Click the "Analyze Paper" button and wait for the report to be generated.
    - Once complete, click the "Export Citation (.bib)" button to get the citation.

2.  **To Explore a Topic:**
    - Select the **"Explore a Research Topic"** tab.
    - Type your area of interest into the text box.
    - Click the "Explore Topic" button.
    - The Scout Agent will perform a search and display its findings.

---


## 📂 File Structure

-   `app.py`: The main application file containing the Gradio UI and the core control flow.
-   `analysis.py`: Defines the deep-dive analysis workflow for a single paper.
-   `agents.py`: Contains the system prompts and creation logic for the AI agents.
-   `utils.py`: Includes helper functions for tasks like fetching the LLM client, downloading PDFs, and formatting BibTeX citations.
-   `requirements.txt`: A list of all the Python packages required to run the project.

## 🔮 Future Scope and Intended Development

This project serves as a strong foundation for a more comprehensive research suite. The following features are planned for future development:

### 1. Interactive RAG Chat Engine
- **Goal:** To move beyond a static analysis report and enable a fully interactive dialogue.
- **Functionality:** After a paper is analyzed, the user will be presented with a chat interface. They can ask follow-up questions ("Can you explain the methodology in simpler terms?"), request clarifications ("What does the ablation study for component X show?"), and have a natural conversation about the paper's content.
- **Implementation:** This involves replacing the `QueryEngine` with a `ChatEngine` in `analysis.py` and updating the Gradio UI in `app.py` to support a conversational format.

### 2. "Download as PDF" Tool
- **Goal:** Allow users to save and share the generated AI analysis.
- **Functionality:** A "Download Report" button will appear alongside the analysis. Clicking it will convert the generated markdown report into a clean, professionally formatted PDF document.
- **Implementation:** This requires adding the `markdown2` and `weasyprint` libraries and creating a new utility function to handle the conversion from markdown to HTML and then to PDF.

### 3. Proactive Discovery with ArXiv Daily Digest
- **Goal:** Transform the application from a reactive tool to a proactive discovery engine.
- **Functionality:** A new "Daily Digest" tab will be added. This tool will automatically fetch the latest preprints from a user-selected arXiv category (e.g., `cs.AI` or `cs.LG`) and display their titles, authors, and abstracts.
- **Implementation:** This involves using the `arxiv` Python library to query the arXiv API directly, without the need for an LLM call for the initial fetch.

### 4. Advanced Agent: The "Peer Reviewer"
- **Goal:** Provide constructive criticism on a user's own work.
- **Functionality:** A new "Critique My Draft" tab will allow users to upload their own draft papers. A team of specialized "critic" agents will then analyze the draft for clarity, methodological soundness, and the strength of its arguments, providing valuable feedback.
- **Implementation:** This would involve creating a new set of system prompts in `agents.py` focused on critique and feedback, and a new workflow to manage this analysis.