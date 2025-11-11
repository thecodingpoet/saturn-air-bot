import json
import logging

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

MODEL = "gpt-4.1-nano"

EVALUATION_PROMPT = """
You are a RAG system quality evaluator for Saturn Airlines Q&A Assistant.

Evaluate the provided answer on these criteria (0-10 scale for each):

1. **Chunk Relevance (0-10)**: How relevant are the retrieved chunks to the user's question?
   - 10: All chunks directly address the question
   - 5: Some chunks are relevant, others tangential
   - 0: Chunks are completely irrelevant or question is outside scope

2. **Answer Accuracy (0-10)**: Does the answer match the information in the chunks?
   - 10: Answer perfectly matches chunk content, no hallucinations
   - 5: Answer mostly accurate but has minor inconsistencies
   - 0: Answer contradicts chunks or contains false information/hallucinations

3. **Completeness (0-10)**: Does the answer fully address the user's question?
   - 10: Comprehensive answer covering all aspects
   - 5: Answer covers main points but misses details
   - 0: Answer is incomplete or evasive
   - SPECIAL CASE: If question is out of scope, a proper decline response is considered complete:
     * For unrelated questions (general knowledge, other companies): Score 8-10 if politely declines
     * For Saturn Airlines-related questions when retrieved chunks don't contain relevant information: Score 8-10 if declines with contact information

4. **Tone & Professionalism (0-10)**: Is the response appropriate in tone?
   - 10: Friendly, professional, and empathetic
   - 5: Acceptable but could be more personable
   - 0: Inappropriate or unprofessional tone

5. **Out-of-Scope Handling (0-10)**: If the retrieved chunks are irrelevant (question is outside scope or information not found in chunks), does the answer properly decline?
   - DISTINGUISH between two types of out-of-scope questions:
   
   a) **Completely unrelated questions** (general knowledge, other companies, unrelated topics):
      - 10: Politely declines without contact info (appropriate for unrelated topics)
      - 7: Declines but awkwardly includes contact info when not needed
      - 5: Attempts to decline but unclear
      - 0: Answers incorrectly, hallucinates, or fails to decline
   
   b) **Saturn Airlines-related questions when retrieved chunks don't contain relevant information** (booking issues, account problems, complex policies):
      - 10: Politely declines AND provides contact information (1-800-SATURN-1 or customerservice@saturnairlines.com)
      - 7: Declines but contact info is incomplete or unclear
      - 5: Declines without contact info when it should be provided
      - 0: Answers incorrectly, hallucinates, or fails to decline
   
   - N/A (score 10): If question is within scope and answered correctly, this criterion is automatically satisfied

IMPORTANT: If chunk_relevance_score is 0-2 (chunks are irrelevant), evaluate based on question type:
- **Unrelated questions** (e.g., "What is the capital of France?"): Answer should politely decline without contact info
- **Saturn Airlines-related questions** (e.g., "How do I change my booking?" when retrieved chunks don't contain relevant information): Answer should decline WITH contact information
- **NEVER** make up or hallucinate information regardless of question type

Return your evaluation as JSON with this structure:
{{
    "chunk_relevance_score": <0-10>,
    "answer_accuracy_score": <0-10>,
    "completeness_score": <0-10>,
    "tone_score": <0-10>,
    "out_of_scope_handling_score": <0-10>,
    "overall_score": <0-10>,
    "strengths": ["..."],
    "weaknesses": ["..."],
    "improvement_suggestions": "..."
}}

User Question: {query}
Retrieved Chunks: {chunks}
Assistant Answer: {answer}
"""

load_dotenv()

evaluator_llm = ChatOpenAI(model_name=MODEL, temperature=0)


def evaluate_answer(query: str, answer: str, chunks: list[Document]) -> dict:
    """
    Evaluate answer quality based on relevance, accuracy, and completeness.
    """
    chunks_text = "\n\n---\n".join(
        [f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(chunks)]
    )

    prompt = EVALUATION_PROMPT.format(query=query, chunks=chunks_text, answer=answer)

    logging.info("📊 Evaluating answer quality...")
    response = evaluator_llm.invoke([HumanMessage(content=prompt)])

    try:
        evaluation = json.loads(response.content)
        return evaluation
    except json.JSONDecodeError:
        logging.error("Failed to parse evaluation response")
        return {"error": "Could not parse evaluation"}
