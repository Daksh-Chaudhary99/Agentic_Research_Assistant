import os
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec

# --- Agent Personas (System Prompts) ---

METHODOLOGY_PROMPT = """You are an expert technical communicator and teacher. 
Your goal is to explain the paper's methodology not just as a list, but as a logical story. Narrate the journey from the problem the authors faced to the solution they designed. 
For each major step or technique (like SBFL or manual inspection), explain its purpose (the 'why') and how it connects to the overall goal of the research.
Use analogies if they help clarify complex concepts. The reader should feel like they understand how the system works, not just what it is made of.
**You must use the `research_paper_query_tool` to find the relevant information from the document to construct your answer.**
"""

RESULTS_PROMPT = """You are a sharp and insightful data analyst. Your task is not just to report the numbers from the 'Results' section, but to interpret them.
Present the key results, using tables if appropriate. Then, immediately after, add a section called "Key Takeaways".
In this section, explain what these results signify in plain language. Is the performance good? Is it surprising? How do these results support or challenge the paper's main claims?
**You must use the `research_paper_query_tool` to find the relevant information from the document to construct your answer.**
"""

CITATION_PROMPT = """You are a historian of science and a domain expert in this field. Your goal is to explain the intellectual lineage of this paper.
Identify the 2-3 most critical papers or concepts cited in the 'Introduction' and 'Related Work' sections.
For each, explain the core idea it introduced and, most importantly, how the current paper **builds upon, challenges, or extends** that foundational work. Help the reader understand where this paper fits in the broader scientific conversation.
**You must use the `research_paper_query_tool` to find the relevant information from the document to construct your answer.**
"""

FUTURE_WORK_PROMPT = """You are an innovative and forward-thinking research strategist. Your task is to look beyond the paper and brainstorm concrete, actionable next steps.
Based on the paper's 'Conclusion' and 'Future Work' sections, as well as any limitations you can infer, generate a list of 3-4 'Future Research Trajectories'.
For each trajectory, provide:
1.  **A clear Research Question:** Phrase it as a question (e.g., "How can we adapt this model to handle real-time code streams?").
2.  **A Proposed First Step:** Suggest a tangible experiment, dataset to use, or architectural change to investigate the question.
3.  **Potential Impact:** Briefly explain why answering this question would be important for the field.
**You must use the `research_paper_query_tool` to find the relevant information from the document to construct your answer.**
"""

SCOUT_PROMPT = """You are a highly skilled research scout for an AI research team.
Your sole purpose is to find the most relevant and recent research papers based on a user's query.
You must use your search tool to find 2 of the most relevant papers from the last 2-3 years, prioritizing sources like arxiv.org.

**If the user's query is very broad (e.g., 'Software Engineering', 'Machine Learning'), you should refine your search to look for 'survey papers' or 'review articles' on that topic.**

You MUST return ONLY a list of direct links to the PDF versions of these papers, separated by newlines. Do not add any commentary or explanation.

Example Output:
https://arxiv.org/pdf/2305.12345.pdf
https://arxiv.org/pdf/2401.54321.pdf
"""

# --- Agent Creation Functions ---

def get_query_tool(index):
    """Provides the tool to query the knowledge base."""
    query_engine = index.as_query_engine(similarity_top_k=3)
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="research_paper_query_tool",
            description="A tool for querying specific information from the indexed research paper.",
        ),
    )
    return query_tool

def create_specialist_agent(system_prompt: str, llm, query_tool, verbose=False):
    """Factory to create a specialist agent with a specific role."""
    return ReActAgent.from_tools(
        tools=[query_tool],
        llm=llm,
        system_prompt=system_prompt,
        verbose=verbose
    )

def create_scout_agent(llm, verbose=True):
    """Creates the agent responsible for finding papers."""
    
    # 1. Read the API key securely from the environment variables
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    # 2. Add a check to ensure the key was found
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set. Please add it to your Space secrets.")
        
    # 3. Use the variable to initialize the tool
    tavily_tool_spec = TavilyToolSpec(api_key=tavily_api_key)
    
    agent = ReActAgent.from_tools(
        tools=tavily_tool_spec.to_tool_list(),
        llm=llm,
        system_prompt=SCOUT_PROMPT,
        verbose=verbose
    )
    return agent