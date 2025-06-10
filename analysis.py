from llama_index.core import VectorStoreIndex, Settings

COMPREHENSIVE_ANALYSIS_PROMPT = """
Your task is to act as an expert research analyst and create a technical brief of the provided document. The primary goal is to make the paper's core motivation, methodology, and findings quickly understandable to another researcher or graduate student in the field.

Provide a comprehensive technical analysis by structuring your response in Markdown with the following sections, in this exact order:

## 1. The Problem & The Core Idea
(Start by clearly stating the specific problem the authors are trying to solve. Explain the context: Why is this problem important or difficult? What are the existing challenges or gaps the authors identify? Then, in a single, clear sentence, state the paper's core proposed solution or main contribution.)

## 2. Core Architecture and Methodology
(Deconstruct the system's architecture and the flow of data or logic. Use bullet points to detail key components, algorithms, and data sources used. Be technically precise and explain the purpose of each major component.)

## 3. Quantitative Results & Critical Analysis
(Present the main quantitative results in a responsive list or table format. Provide a critical analysis of what these results signify. Do they strongly support the main hypothesis? Mention any important ablation studies or comparisons.)

## 4. Positioning in the Field
(Situate this work by comparing it to 1-2 key alternative approaches mentioned in the paper. Clearly articulate how this work's architecture or methodology differentiates itself and what the trade-offs are.)

## 5. Proposed Future Research Directions
(Propose 2-3 concrete, technically-grounded hypotheses for extending this research. For each, describe a potential experiment or technical extension.)
"""

# # The master prompt that asks for everything at once.
# COMPREHENSIVE_ANALYSIS_PROMPT = """
# Provide a comprehensive technical analysis of the document for a knowledgeable audience (e.g., graduate students, researchers). Structure your response in Markdown with the following sections, in this exact order:

# ## 1. Abstract Summary
# (A concise summary of the paper's core contributions, methods, and key results, similar to a conference abstract.)

# ## 2. Core Architecture and Methodology
# (Deconstruct the system's architecture and the flow of data or logic. Use bullet points to detail key components and algorithms. Be technically precise.)

# ## 3. Quantitative Results & Critical Analysis
# (Present the main quantitative results in a list or responsive format (NO WIDE TABLES). Provide a brief but critical analysis of what these results mean.)

# ## 4. Positioning in the Field
# (Situate this work by comparing it to 1-2 key alternative approaches mentioned in the paper, highlighting its unique technical differentiators.)

# ## 5. Proposed Future Research Directions
# (Propose 2-3 concrete, technically-grounded hypotheses and experimental ideas for extending this research based on the paper's conclusion or limitations.)
# """

def run_analysis_on_single_paper(documents):
    """
    This simplified version creates an index and runs a single, comprehensive query against it.
    """
    print("--- Simplified Analysis: Indexing the document... ---")
    # This creates the index in memory for analysis
    index = VectorStoreIndex.from_documents(documents)
    
    print("--- Simplified Analysis: Creating Query Engine... ---")
    # A query engine is a stable way to ask questions about the document
    query_engine = index.as_query_engine(
        response_mode="tree_summarize",  # This mode is excellent for summarizing a whole document
    )

    print("--- Simplified Analysis: Running comprehensive query... ---")
    # We run our one big prompt against the entire document.
    final_report = query_engine.query(COMPREHENSIVE_ANALYSIS_PROMPT)
    
    print("--- Simplified Analysis: Analysis complete. ---")
    return str(final_report)