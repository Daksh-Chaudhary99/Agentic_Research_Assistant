# app.py (Simplified "Explore Topic" tab)

import gradio as gr
import os
import re
import hashlib
from llama_index.core import Settings, Document
from llama_index.readers.file import PDFReader
from llama_index.embeddings.mistralai import MistralAIEmbedding

# Import our custom modules
from utils import get_llm, download_pdf_from_url, format_to_bibtex
from agents import create_scout_agent, create_specialist_agent, CITATION_EXTRACTOR_PROMPT
from analysis import run_analysis_on_single_paper

# --- Orchestrator Functions for Gradio ---

def pdf_analysis_flow(pdf_file, progress=gr.Progress()):
    """Workflow for the 'Analyze a Specific PDF' tab."""
    if pdf_file is None:
        raise gr.Error("Please upload a PDF file.")
        
    try:
        progress(0.2, desc="Setting up AI models...")
        Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")
        Settings.llm = get_llm()

        progress(0.5, desc="Analyzing paper...")
        documents = PDFReader().load_data(file=pdf_file.name)
        report_title = f"# Analysis of: *{os.path.basename(pdf_file.name)}*\n\n"
        
        final_report = run_analysis_on_single_paper(documents)
        
        return report_title + final_report, documents, gr.update(visible=True)
    except Exception as e:
        print(f"An error occurred in pdf_analysis_flow: {e}")
        return f"An error occurred: {e}", None, gr.update(visible=False)

def export_bibtex_flow(documents, file_obj):
    """Workflow for the 'Export Citation' button."""
    if not documents:
        raise gr.Error("Please analyze a paper first.")
    
    filename = os.path.basename(file_obj.name)
    print(f"--- BibTeX Export: Starting citation extraction for {filename} ---")
    
    first_page_text = documents[0].text
    Settings.llm = get_llm()
    extractor_agent = create_specialist_agent(CITATION_EXTRACTOR_PROMPT, Settings.llm, [])
    
    response = extractor_agent.chat(f"Extract bibliographic data from this text: {first_page_text[:4000]}")
    
    print(f"--- BibTeX Export: Agent responded with: {response.response} ---")
    bibtex_string = format_to_bibtex(response.response, filename)
    
    return bibtex_string

def scout_agent_flow(topic_query, progress=gr.Progress()):
    """This function now runs the scout agent and directly returns its summary."""
    if not topic_query:
        raise gr.Error("Please enter a research topic.")

    progress(0, desc="Setting up AI model...")
    Settings.llm = get_llm()
    
    progress(0.3, desc="Scout Agent is searching for relevant papers...")
    formatted_query = f"{topic_query} site:arxiv.org"
    
    scout_agent = create_scout_agent(Settings.llm, verbose=True)
    response = scout_agent.chat(formatted_query)
    
    # MODIFIED: Now only returns a single string for the markdown output
    return str(response)

# --- Gradio UI Definition ---

with gr.Blocks(theme=gr.themes.Soft(), title="AI Research Assistant") as demo:
    gr.Markdown("# 🤖 AI Research Assistant")
    gr.Markdown("Your AI-powered partner for literature discovery and analysis, powered by Mistral.")
    
    document_state = gr.State()
    
    with gr.Tabs():
        with gr.TabItem("Analyze a Specific PDF"):
            with gr.Column():
                pdf_input = gr.File(type="filepath", label="Upload Research Paper (PDF)")
                analyze_button_pdf = gr.Button("Analyze Paper", variant="primary")
                pdf_output = gr.Markdown(label="Analysis Report")

                with gr.Group(visible=False) as tools_group:
                    gr.Markdown("### 🛠️ Tools")
                    export_bibtex_button = gr.Button("Export Citation (.bib)")
                    bibtex_output = gr.Textbox(
                        label="BibTeX Citation", 
                        show_copy_button=True, 
                        interactive=False,
                        lines=7
                    )
        
        # MODIFIED: This tab is now simplified
        with gr.TabItem("Explore a Research Topic"):
            with gr.Column():
                topic_input = gr.Textbox(lines=3, label="Enter your Research Topic or Idea")
                explore_button = gr.Button("Explore Topic", variant="primary")
                scout_results_display = gr.Markdown(label="Scout Agent Findings")

    # Wiring for the "Analyze a Specific PDF" tab
    analyze_button_pdf.click(
        fn=pdf_analysis_flow, 
        inputs=[pdf_input], 
        outputs=[pdf_output, document_state, tools_group]
    )
    export_bibtex_button.click(
        fn=export_bibtex_flow,
        inputs=[document_state, pdf_input],
        outputs=[bibtex_output]
    )
    
    # MODIFIED: Wiring for the simplified "Explore a Research Topic" tab
    explore_button.click(
        fn=scout_agent_flow,
        inputs=[topic_input],
        outputs=[scout_results_display]
    )

if __name__ == "__main__":
    demo.launch()