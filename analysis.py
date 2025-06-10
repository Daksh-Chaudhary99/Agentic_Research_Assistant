from llama_index.core import VectorStoreIndex, Settings

# The master prompt that asks for everything at once.
COMPREHENSIVE_ANALYSIS_PROMPT = """
Provide a comprehensive technical analysis of the document for a knowledgeable audience (e.g., graduate students, researchers). Structure your response in Markdown with the following sections, in this exact order:

## 1. Abstract Summary
(A concise summary of the paper's core contributions, methods, and key results, similar to a conference abstract.)

## 2. Core Architecture and Methodology
(Deconstruct the system's architecture and the flow of data or logic. Use bullet points to detail key components and algorithms. Be technically precise.)

## 3. Quantitative Results & Critical Analysis
(Present the main quantitative results in a list or responsive format (NO WIDE TABLES). Provide a brief but critical analysis of what these results mean.)

## 4. Positioning in the Field
(Situate this work by comparing it to 1-2 key alternative approaches mentioned in the paper, highlighting its unique technical differentiators.)

## 5. Proposed Future Research Directions
(Propose 2-3 concrete, technically-grounded hypotheses and experimental ideas for extending this research based on the paper's conclusion or limitations.)
"""

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