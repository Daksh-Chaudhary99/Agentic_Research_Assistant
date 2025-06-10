import gradio as gr
import os
from llama_index.core import Settings
from llama_index.readers.file import PDFReader
from llama_index.embeddings.mistralai import MistralAIEmbedding
from utils import get_llm, download_pdf_from_url
from agents import create_scout_agent
from analysis import run_analysis_on_single_paper

# --- Orchestrator Functions for Gradio ---

def pdf_analysis_flow(pdf_file):
    """Orchestrator for the PDF upload workflow."""
    if pdf_file is None:
        return "Error: Please upload a PDF file."
        
    try:
        # --- Set the models globally for LlamaIndex ---
        # Mistral's embedding API
        Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")
        # Mistral's LLM for thinking/writing
        Settings.llm = get_llm()

        documents = PDFReader().load_data(file=pdf_file.name)
        report_title = f"# Analysis of: *{os.path.basename(pdf_file.name)}*\n\n"
        final_report = run_analysis_on_single_paper(documents)
        return report_title + final_report
    except Exception as e:
        print(f"An error occurred in pdf_analysis_flow: {e}")
        return f"An error occurred: {e}"


def topic_exploration_flow(topic_query):
    """Orchestrator for the topic exploration workflow."""
    if not topic_query:
        return "Error: Please enter a research topic."

    try:
        # --- Set the models globally for LlamaIndex ---
        Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")
        Settings.llm = get_llm()

        # 1. Run Scout Agent to get paper URLs
        gr.Info("Scout Agent is searching for relevant papers...")
        
        # We automatically focus the search on arXiv for better results.
        formatted_query = f"{topic_query} site:arxiv.org"
        print(f"DEBUGGING: Sending formatted query to scout: '{formatted_query}'")
        
        scout_agent = create_scout_agent(Settings.llm)
        # We pass the more specific query to the agent.
        response = scout_agent.chat(formatted_query)
                
        # --- DEBUGGING CODE ---
        print("--- RAW AGENT RESPONSE ---")
        print(response)
        print("--- END RAW AGENT RESPONSE ---")
        # --- DEBUGGING CODE ---
        
        urls = [line.strip() for line in response.response.split('\n') if line.strip().startswith('http')]

        if not urls:
            return "Scout Agent could not find any relevant papers. Please try a different query."

        # The rest of the function remains the same...
        # 2. Loop through URLs and analyze each paper
        all_reports = []
        for i, url in enumerate(urls):
            gr.Info(f"Analyzing paper {i+1}/{len(urls)}: {url}")
            pdf_stream = download_pdf_from_url(url)
            if pdf_stream:
                documents = PDFReader().load_data(file=pdf_stream)
                report_title = f"# Analysis of Paper from: *{url}*\n\n"
                single_report = run_analysis_on_single_paper(documents)
                all_reports.append(report_title + single_report)
            else:
                all_reports.append(f"## Could not analyze paper from {url}\n\nFailed to download or process the PDF.")

        # 3. Combine all reports
        gr.Info("Combining all reports...")
        final_combined_report = f"# Research Exploration on: '{topic_query}'\n\n"
        final_combined_report += "The Scout Agent found the following papers, and the Research Team has analyzed them below:\n\n---\n\n"
        final_combined_report += "\n\n---\n\n".join(all_reports)
        
        return final_combined_report
    except Exception as e:
        print(f"An error occurred in topic_exploration_flow: {e}")
        return f"An error occurred: {e}"

# --- Gradio UI Definition ---

with gr.Blocks(theme=gr.themes.Soft(), title="AI Research Assistant") as demo:
    gr.Markdown("# 🤖 AI Research Assistant")
    gr.Markdown("Your AI-powered partner for literature discovery and analysis, powered by Mistral.")
    
    with gr.Tabs():
        with gr.TabItem("Analyze a Specific PDF"):
            pdf_input = gr.File(type="filepath", label="Upload Research Paper (PDF)")
            analyze_button_pdf = gr.Button("Analyze Paper", variant="primary")
            pdf_output = gr.Markdown(label="Analysis Report")

        with gr.TabItem("Explore a Research Topic"):
            topic_input = gr.Textbox(lines=3, label="Enter your Research Topic or Idea")
            analyze_button_topic = gr.Button("Explore Topic", variant="primary")
            topic_output = gr.Markdown(label="Combined Analysis Report")

    analyze_button_pdf.click(
        fn=pdf_analysis_flow, 
        inputs=[pdf_input], 
        outputs=pdf_output
    )
    analyze_button_topic.click(
        fn=topic_exploration_flow, 
        inputs=[topic_input], 
        outputs=topic_output
    )

if __name__ == "__main__":
    demo.launch()