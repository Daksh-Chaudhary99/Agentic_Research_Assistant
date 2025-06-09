import concurrent.futures
from llama_index.core import VectorStoreIndex, Settings
from agents import (
    get_query_tool, 
    create_specialist_agent, 
    METHODOLOGY_PROMPT, 
    RESULTS_PROMPT, 
    CITATION_PROMPT, 
    FUTURE_WORK_PROMPT
)

def run_analysis_on_single_paper(documents, llm):
    """
    Orchestrates the multi-agent analysis for a single paper.
    
    Args:
        documents (list): A list of LlamaIndex Document objects.
        llm (LLM): The LLM instance to use for the analysis.
        
    Returns:
        str: A formatted markdown report of the analysis.
    """
    Settings.llm = llm

    # Step 1: Librarian Agent builds the knowledge base
    print("Librarian Agent: Indexing the document...")
    index = VectorStoreIndex.from_documents(documents)
    query_tool = get_query_tool(index)
    print("Librarian Agent: Knowledge base is ready.")

    # Step 2: Assemble the specialist team
    specialists = {
        "Methodology": create_specialist_agent(METHODOLOGY_PROMPT, llm, query_tool),
        "Results": create_specialist_agent(RESULTS_PROMPT, llm, query_tool),
        "Citations": create_specialist_agent(CITATION_PROMPT, llm, query_tool),
        "Future Work": create_specialist_agent(FUTURE_WORK_PROMPT, llm, query_tool),
    }

    # Step 3: Run specialists in parallel
    print("Lead Researcher: Delegating tasks to specialists in parallel...")
    individual_reports = {}
    tasks = {
        "Methodology": "Analyze the methodology in detail.",
        "Results": "Summarize the key results and performance metrics.",
        "Citations": "Identify and explain the importance of foundational citations.",
        "Future Work": "List the identified research gaps and future work.",
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_role = {executor.submit(agent.chat, tasks[role]): role for role, agent in specialists.items()}
        for future in concurrent.futures.as_completed(future_to_role):
            role = future_to_role[future]
            try:
                result = future.result()
                individual_reports[role] = result.response
                print(f"Lead Researcher: Received report from {role} Analyst.")
            except Exception as exc:
                individual_reports[role] = f"Error processing {role}: {exc}"

    # Step 4: Synthesize the final report
    print("Lead Researcher: Synthesizing final report...")
    synthesis_prompt = f"""You are the Lead Researcher. You have received reports from your specialist team.
    Your job is to synthesize these into a single, cohesive, and well-structured final analysis.
    The report should be accessible to a technical audience. Format it in markdown.

    --- Methodology Report ---
    {individual_reports.get('Methodology', 'N/A')}

    --- Results Report ---
    {individual_reports.get('Results', 'N/A')}

    --- Foundational Citations Report ---
    {individual_reports.get('Citations', 'N/A')}

    --- Future Work Report ---
    {individual_reports.get('Future Work', 'N/A')}
    ---

    Synthesize the final report now. Start with a high-level summary, then detail each section.
    """
    
    final_report = llm.complete(synthesis_prompt)
    return final_report.text