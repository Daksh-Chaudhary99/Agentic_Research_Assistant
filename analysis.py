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


def run_analysis_on_single_paper(documents):
    """
    Orchestrates the multi-agent analysis for a single paper.
    It now uses the globally set LLM from Settings.
    """
    
    # Step 1: Librarian Agent builds the knowledge base (uses the global embed_model)
    print("Librarian Agent: Indexing the document...")
    index = VectorStoreIndex.from_documents(documents)
    query_tool = get_query_tool(index)
    print("Librarian Agent: Knowledge base is ready.")

    # Step 2: Assemble the specialist team, using the global Settings.llm
    specialists = {
        "Methodology": create_specialist_agent(METHODOLOGY_PROMPT, Settings.llm, query_tool),
        "Results": create_specialist_agent(RESULTS_PROMPT, Settings.llm, query_tool),
        "Citations": create_specialist_agent(CITATION_PROMPT, Settings.llm, query_tool),
        "Future Work": create_specialist_agent(FUTURE_WORK_PROMPT, Settings.llm, query_tool),
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

    # Step 4: Synthesize the final report using the global Settings.llm
    print("Lead Researcher: Synthesizing final report...")

    synthesis_prompt = f"""You are a master science communicator and Lead Researcher. Your task is to synthesize the detailed reports from your specialist team into a single, polished, and highly intuitive final report. The goal is for someone to understand the paper's essence and potential on a first read.

    Structure the final report in Markdown exactly as follows:

    # [Paper Title - Generate a fitting title based on the content]

    ## The Core Idea in a Nutshell
    (Provide a one-paragraph, easy-to-understand summary of what this paper is about and why it matters. Use an analogy if possible.)

    ---

    ## How It Works: The Methodology Explained
    (Synthesize the Methodology Analyst's report here. Ensure it flows like a story, explaining the 'why' behind each step.)

    ---

    ## What They Found: Results and Key Takeaways
    (Synthesize the Results Analyst's report. Present the key data and, most importantly, the interpretation and takeaways.)

    ---

    ## The Scientific Context: Foundational Work
    (Synthesize the Citation Analyst's report. Explain the key prior works and how this paper connects to them.)

    ---

    ## Where to Go From Here: Future Research Trajectories
    (Synthesize the Future-Work Scout's report. Present the actionable next steps as clear, distinct ideas.)

    ---
    
    **Specialist Reports Used for this Synthesis:**

    **Methodology Report:**
    {individual_reports.get('Methodology', 'N/A')}

    **Results Report:**
    {individual_reports.get('Results', 'N/A')}

    **Foundational Citations Report:**
    {individual_reports.get('Citations', 'N/A')}

    **Future Work Report:**
    {individual_reports.get('Future Work', 'N/A')}
    ---

    Now, generate the final, synthesized report based on these instructions.
    """
    
    final_report = Settings.llm.complete(synthesis_prompt)
    return final_report.text