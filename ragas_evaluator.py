from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate, EvaluationDataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """
    Evaluate a (question, context, answer) triple using RAGAS metrics.

    Args:
        question: The user question.
        answer: The model-generated answer.
        contexts: List of retrieved context strings.

    Returns:
        Dictionary keyed by metric name with float scores, or an error dict.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available. Install with: pip install ragas"}

    # Input validation
    if not question or not isinstance(question, str) or not question.strip():
        return {"error": "Question must be a non-empty string."}
    if not answer or not isinstance(answer, str) or not answer.strip():
        return {"error": "Answer must be a non-empty string."}
    if not contexts or not isinstance(contexts, list) or not any(c.strip() for c in contexts if isinstance(c, str)):
        return {"error": "Contexts must be a non-empty list of non-empty strings."}

    # Filter out empty context strings
    contexts = [c for c in contexts if isinstance(c, str) and c.strip()]

    import os
    base_url = os.environ.get("OPENAI_API_BASE")

    # Create evaluator LLM
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", base_url=base_url))

    # Create evaluator embeddings
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", base_url=base_url)
    )

    # Define metrics
    metrics = [
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm),
        NonLLMContextPrecisionWithReference(),
        BleuScore(),
        RougeScore(),
    ]

    # Build evaluation sample
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference="\n".join(contexts),
        reference_contexts=contexts
    )

    # Build evaluation dataset and run evaluation
    eval_dataset = EvaluationDataset(samples=[sample])

    eval_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return eval_result.scores[0]
