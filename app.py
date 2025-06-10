# app.py (Corrected to use run_analysis_on_single_paper)

import gradio as gr
import os
import re
from llama_index.core import Settings
from llama_index.readers.file import PDFReader
from llama_index.embeddings.mistralai import MistralAIEmbedding
from utils import get_llm, download_pdf_from_url
from agents import create_scout_agent
from analysis import run_analysis_on_single_paper

# --- Orchestrator Functions for Gradio ---

def pdf_analysis_flow(pdf_file, progress=gr.Progress()):
    """A simple workflow for the 'Analyze a Specific PDF' tab."""
    if pdf_file is None:
        raise gr.Error("Please upload a PDF file.")
        
    try:
        progress(0.2, desc="Setting up AI models...")
        Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")
        Settings.llm = get_llm()

        progress(0.5, desc="Analyzing paper...")
        documents = PDFReader().load_data(file=pdf_file.name)
        report_title = f"# Analysis of: *{os.path.basename(pdf_file.name)}*\n\n"
        
        # Call the original analysis function
        final_report = run_analysis_on_single_paper(documents)
        
        return report_title + final_report
    except Exception as e:
        print(f"An error occurred in pdf_analysis_flow: {e}")
        return f"An error occurred: {e}"


def scout_agent_flow(topic_query, progress=gr.Progress()):
    """Step 1 for the Explore tab: Runs ONLY the scout agent."""
    if not topic_query:
        raise gr.Error("Please enter a research topic.")

    progress(0, desc="Setting up AI model...")
    Settings.llm = get_llm()
    
    progress(0.3, desc="Scout Agent is searching for relevant papers...")
    formatted_query = f"{topic_query} site:arxiv.org"
    
    scout_agent = create_scout_agent(Settings.llm, verbose=True)
    response = scout_agent.chat(formatted_query)
    
    # Return the agent's raw, helpful response and make the next UI section visible
    return str(response), gr.update(visible=True)


def url_analysis_flow(url_to_analyze, progress=gr.Progress()):
    """Step 2 for the Explore tab: Analyzes a single URL provided by the user."""
    if not url_to_analyze or not url_to_analyze.startswith("http"):
        raise gr.Error("Please enter a valid URL from the list above.")

    try:
        progress(0, desc="Setting up AI models...")
        Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")
        Settings.llm = get_llm()

        progress(0.2, desc=f"Downloading paper from {url_to_analyze}...")
        pdf_url = url_to_analyze.replace("/abs/", "/pdf/") if "arxiv.org/abs" in url_to_analyze else url_to_analyze
        pdf_stream = download_pdf_from_url(pdf_url)

        if not pdf_stream:
            raise gr.Error("Failed to download the PDF from the provided URL.")

        progress(0.5, desc="Analyzing paper...")
        documents = PDFReader().load_data(file=pdf_stream)
        
        # Call the original analysis function
        final_report = run_analysis_on_single_paper(documents)
        
        return final_report
    except Exception as e:
        print(f"An error occurred in url_analysis_flow: {e}")
        return f"An error occurred: {e}"


# --- Gradio UI Definition ---

with gr.Blocks(theme=gr.themes.Soft(), title="AI Research Assistant") as demo:
    gr.Markdown("# 🤖 AI Research Assistant")
    gr.Markdown("Your AI-powered partner for literature discovery and analysis, powered by Mistral.")
    
    with gr.Tabs():
        
        with gr.TabItem("Analyze a Specific PDF"):
            with gr.Column():
                pdf_input = gr.File(type="filepath", label="Upload Research Paper (PDF)")
                analyze_button_pdf = gr.Button("Analyze Paper", variant="primary")
                pdf_output = gr.Markdown(label="Analysis Report")

        # This tab uses the two-step workflow
        with gr.TabItem("Explore a Research Topic"):
            with gr.Column():
                # Step 1 UI
                topic_input = gr.Textbox(lines=3, label="Enter your Research Topic or Idea")
                explore_button = gr.Button("Explore Topic", variant="primary")
                scout_results_display = gr.Markdown(label="Scout Agent Findings")
                
                # Step 2 UI, hidden by default
                with gr.Box(visible=False) as url_analysis_box:
                    gr.Markdown("Copy a URL from the summary above and paste it here for a deep-dive analysis.")
                    url_input_textbox = gr.Textbox(label="Paper URL to Analyze")
                    analyze_url_button = gr.Button("Analyze This Paper", variant="secondary")
                
                single_analysis_display = gr.Markdown(label="Deep-Dive Analysis")

    # Wire up the buttons to the backend functions
    analyze_button_pdf.click(
        fn=pdf_analysis_flow, 
        inputs=[pdf_input], 
        outputs=[pdf_output]
    )
    explore_button.click(
        fn=scout_agent_flow,
        inputs=[topic_input],
        outputs=[scout_results_display, url_analysis_box]
    )
    analyze_url_button.click(
        fn=url_analysis_flow,
        inputs=[url_input_textbox],
        outputs=[single_analysis_display]
    )

if __name__ == "__main__":
    demo.launch()