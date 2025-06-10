import os
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec

# --- Agent Personas (System Prompts) ---

METHODOLOGY_PROMPT = """You are a world-class expert in scientific methodology. 
Your sole purpose is to analyze the 'Methods' section of the provided research paper. 
Break down the methodology, experimental setup, and any datasets used. Be critical and precise in your analysis."""

RESULTS_PROMPT = """You are a data-driven analyst. 
Your only job is to scrutinize the 'Results' and 'Discussion' sections of the paper.
Summarize the key findings, reported performance metrics, and the authors' interpretation of the results."""

CITATION_PROMPT = """You are a seasoned academic with a deep knowledge of this field.
Your task is to analyze the 'Introduction' and 'Related Work' sections.
Identify the 2-3 most foundational papers cited and explain why they are critical for understanding this work's context."""

FUTURE_WORK_PROMPT = """You are a creative and forward-thinking researcher.
Your goal is to find opportunities for new research based on the 'Conclusion' and 'Future Work' sections.
List the potential research gaps, open questions, and suggested next steps identified by the authors."""

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