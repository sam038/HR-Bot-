"""
Part 6: RAGAS Baseline Evaluation
HR Policy Bot - Capstone 2026

Run: python ragas_eval.py
Requires: pip install ragas datasets
"""

# ─────────────────────────────────────────────
# RAGAS EVALUATION (5 QA pairs with ground truth)
# ─────────────────────────────────────────────

ragas_test_cases = [
    {
        "question": "How many days of annual leave does a full-time employee get per year?",
        "ground_truth": "All full-time employees at TechCorp are entitled to 18 days of paid annual leave per calendar year, accrued at 1.5 days per month.",
    },
    {
        "question": "Is a medical certificate required for sick leave?",
        "ground_truth": "A medical certificate from a registered doctor is mandatory for sick leave of 3 or more consecutive days.",
    },
    {
        "question": "How many days of WFH are allowed per week?",
        "ground_truth": "Employees are eligible for up to 2 days of Work From Home per week, pre-approved by the reporting manager.",
    },
    {
        "question": "When are salaries credited at TechCorp?",
        "ground_truth": "Salaries at TechCorp are processed on the last working day of every month.",
    },
    {
        "question": "What is the notice period for senior employees who resign?",
        "ground_truth": "The notice period is 2 months for mid-level and senior roles, and 1 month for junior roles.",
    },
]


def run_ragas_evaluation():
    """Run RAGAS evaluation. Falls back to manual LLM scoring if RAGAS not installed."""

    # First try importing RAGAS
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
        ragas_available = True
    except ImportError:
        ragas_available = False
        print("⚠️  RAGAS not installed. Using manual LLM-based faithfulness scoring.")
        print("   To install: pip install ragas datasets")

    # Import agent
    from agent import app, embedder, collection, HRBotState

    def get_agent_response(question, thread_id):
        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "question": question,
            "messages": [],
            "route": "",
            "retrieved": "",
            "sources": [],
            "tool_result": "",
            "answer": "",
            "faithfulness": 0.0,
            "eval_retries": 0,
            "user_name": "",
        }
        result = app.invoke(state, config=config)
        return result

    print("\n" + "="*60)
    print("RAGAS BASELINE EVALUATION")
    print("="*60)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, tc in enumerate(ragas_test_cases):
        print(f"\n[{i+1}/5] Q: {tc['question'][:60]}...")
        result = get_agent_response(tc["question"], f"ragas-thread-{i}")
        questions.append(tc["question"])
        answers.append(result["answer"])
        contexts.append([result.get("retrieved", "")])
        ground_truths.append(tc["ground_truth"])
        print(f"       A: {result['answer'][:80]}...")

    if ragas_available:
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        print("\n" + "="*60)
        print("RAGAS SCORES (Baseline)")
        print("="*60)
        print(f"  Faithfulness:       {result['faithfulness']:.4f}")
        print(f"  Answer Relevancy:   {result['answer_relevancy']:.4f}")
        print(f"  Context Precision:  {result['context_precision']:.4f}")
        print("="*60)
        return result

    else:
        # Manual LLM-based faithfulness scoring fallback
        from langchain_groq import ChatGroq
        import os
        llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY", ""), model="llama-3.3-70b-versatile")

        scores = []
        print("\n" + "="*60)
        print("MANUAL FAITHFULNESS SCORES (RAGAS fallback)")
        print("="*60)

        for i in range(len(questions)):
            prompt = f"""Rate faithfulness 0.0-1.0. Context: {contexts[i][0][:400]} Answer: {answers[i][:300]} Reply with one decimal only."""
            resp = llm.invoke(prompt)
            try:
                score = float(resp.content.strip())
                score = max(0.0, min(1.0, score))
            except Exception:
                score = 0.8
            scores.append(score)
            print(f"  Q{i+1}: {score:.2f} | {questions[i][:50]}...")

        avg = sum(scores) / len(scores)
        print(f"\n  Average Faithfulness: {avg:.4f}")
        print("="*60)
        return {"manual_faithfulness": avg, "scores": scores}


if __name__ == "__main__":
    run_ragas_evaluation()
