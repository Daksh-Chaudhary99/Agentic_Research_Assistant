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

    synthesis_prompt = f"""You are a Lead Researcher compiling a technical brief for a research group. Synthesize the reports from your specialist agents into a single, structured, and technically deep analysis. The intended audience is graduate students and researchers in the field.

    Structure the final report in Markdown exactly as follows:

    # Technical Analysis: [Paper Title - Generate a fitting title]

    ## 1. Abstract Summary
    (Provide a concise summary of the paper's core contributions, methods, and key results, similar to a conference abstract.)

    ---

    ## 2. Core Architecture and Methodology
    (Synthesize the Methodology Analyst's report. Deconstruct the system's architecture and the flow of data or logic. Use bullet points to detail key components and algorithms.)

    ---

    ## 3. Quantitative Results & Critical Analysis
    (Synthesize the Results Analyst's report. Display the main data in tables. Provide a critical analysis of what these results mean, their statistical significance, and how they support the paper's thesis.)

    ---

    ## 4. Positioning in the Field
    (Synthesize the Citation Analyst's report. Clearly articulate how this work differs from or improves upon specific, named alternative approaches.)

    ---

    ## 5. Proposed Future Research Directions
    (Synthesize the Future-Work Scout's report. Present the concrete, technically-grounded hypotheses and experimental ideas for extending this research.)
    
    ---
    
    **Source Reports from Specialist Agents:**
    (Include the raw reports below for reference)

    **Methodology Report:**
    {individual_reports.get('Methodology', 'N/A')}

    **Results Report:**
    {individual_reports.get('Results', 'N/A')}

    **Citations Report:**
    {individual_reports.get('Citations', 'N/A')}

    **Future Work Report:**
    {individual_reports.get('Future Work', 'N/A')}
    ---

    Generate the final, synthesized report.
    """
    
    final_report = Settings.llm.complete(synthesis_prompt)
    return final_report.text