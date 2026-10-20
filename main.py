import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR / "docs"
DATA_DIR = APP_DIR / "data"
STARTER_DATA_DIR = APP_DIR / "starter_kit" / "data"
QUESTIONS_PATH = STARTER_DATA_DIR / "example_questions.json"
ABSTENTION = "I don't have an answer for that in the documentation."

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours"
}

CONVERSATIONAL_WORDS = {
    "help", "guide", "assist", "assistance", "steps", "tell", "show", "please",
    "want", "need", "know", "way", "problem", "issue", "trouble", "trying",
    "how", "what", "can", "i", "my", "me", "give", "find", "get", "do"
}


def normalize_term(word: str) -> str:
    w = word.lower()
    for sfx in ("ing", "es", "s", "ed", "ly", "tion", "ment"):
        if w.endswith(sfx) and len(w) > len(sfx) + 2:
            return w[:-len(sfx)]
    return w


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    raw_tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    if remove_stopwords:
        tokens = [normalize_term(w) for w in raw_tokens if w not in STOPWORDS]
        return tokens if tokens else [normalize_term(w) for w in raw_tokens]
    return [normalize_term(w) for w in raw_tokens]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75, title_boost: float = 3.0):
        self.k1 = k1
        self.b = b
        self.title_boost = title_boost
        self.corpus: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}
        self.global_vocab: set = set()

    def fit(self, docs: List[Dict[str, Any]]) -> None:
        self.corpus = docs
        self.doc_tokens = []
        self.doc_lens = []
        self.global_vocab = set()

        for doc in docs:
            title_toks = tokenize(doc.get("title", "")) * int(self.title_boost)
            content_toks = tokenize(doc.get("content", ""))
            all_toks = title_toks + content_toks
            self.doc_tokens.append(all_toks)
            self.doc_lens.append(len(all_toks))
            self.global_vocab.update(all_toks)

        num_docs = len(docs)
        self.avg_doc_len = sum(self.doc_lens) / num_docs if num_docs > 0 else 0.0

        doc_frequencies: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                doc_frequencies[term] = doc_frequencies.get(term, 0) + 1

        self.idf = {}
        for term, freq in doc_frequencies.items():
            self.idf[term] = math.log(1 + (num_docs - freq + 0.5) / (freq + 0.5))

    def score_query(self, query: str) -> List[Tuple[float, Dict[str, Any]]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return [(0.0, doc) for doc in self.corpus]

        scores = []
        for i, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_lens[i]
            tokens = self.doc_tokens[i]
            t_counts: Dict[str, int] = {}
            for t in tokens:
                t_counts[t] = t_counts.get(t, 0) + 1

            for qt in q_tokens:
                if qt not in self.idf:
                    continue
                freq = t_counts.get(qt, 0)
                if freq == 0:
                    continue
                idf = self.idf[qt]
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (num / den)

            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores


def load_documents() -> List[Dict[str, str]]:
    docs = []
    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            doc_id = ""
            match_id = re.search(r"^Document ID:\s*(\S+)\s*$", text, re.MULTILINE)
            if match_id:
                doc_id = match_id.group(1)

            title = path.stem.replace("-", " ").replace("_", " ").title()
            match_title = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            if match_title:
                title = match_title.group(1).strip()

            clean_content = re.sub(r"^Document ID:.*$", "", text, flags=re.MULTILINE).strip()
            docs.append({
                "doc_id": doc_id,
                "filename": path.name,
                "title": title,
                "content": clean_content
            })

    if not docs:
        json_paths = []
        for candidate_dir in [DOCS_DIR, DATA_DIR, STARTER_DATA_DIR, APP_DIR]:
            if candidate_dir.exists():
                json_paths.extend(candidate_dir.glob("docs.json"))

        for j_path in json_paths:
            try:
                with j_path.open(encoding="utf-8") as f:
                    data = json.load(f)
                items = data.get("docs", data) if isinstance(data, dict) else data
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            docs.append({
                                "doc_id": item.get("doc_id") or item.get("id", ""),
                                "filename": j_path.name,
                                "title": item.get("title", ""),
                                "content": item.get("content", "")
                            })
                    if docs:
                        break
            except Exception:
                continue

    return docs


def find_questions_file() -> Optional[Path]:
    candidates = [
        QUESTIONS_PATH,
        DATA_DIR / "example_questions.json",
        APP_DIR / "data" / "example_questions.json",
        APP_DIR / "example_questions.json"
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


class DocsQAAgent:
    CONFIDENCE_THRESHOLD = 1.6

    def __init__(self):
        self.docs = load_documents()
        if not self.docs:
            raise RuntimeError("No documentation files found in docs/ or data/docs.json.")
        self.retriever = BM25Retriever()
        self.retriever.fit(self.docs)

    def _extract_best_passage(self, query: str, content: str, max_sentences: int = 2) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        if not sentences:
            return content

        q_tokens = set(tokenize(query))
        scored = []
        for idx, sentence in enumerate(sentences):
            s_toks = tokenize(sentence)
            if not s_toks:
                continue
            overlap = sum(1 for t in s_toks if t in q_tokens)
            score = overlap / (len(s_toks) ** 0.5)
            scored.append((score, idx, sentence))

        if not scored:
            return sentences[0]

        scored.sort(key=lambda x: x[0], reverse=True)
        top_indices = sorted([s[1] for s in scored[:max_sentences]])
        return " ".join([sentences[i] for i in top_indices])

    def answer_question(self, question: str) -> Tuple[str, List[str], Optional[str]]:
        raw_words = re.findall(r"\b[a-z0-9]+\b", question.lower())
        if not raw_words:
            return ABSTENTION, [], None

        norm_tokens = [normalize_term(w) for w in raw_words if w not in STOPWORDS]
        test_tokens = [t for t in norm_tokens if t not in CONVERSATIONAL_WORDS]

        if test_tokens:
            unknown_entities = [t for t in test_tokens if t not in self.retriever.global_vocab]
            if unknown_entities:
                return ABSTENTION, [], None

        results = self.retriever.score_query(question)
        if not results:
            return ABSTENTION, [], None

        top_score, top_doc = results[0]

        if top_score < self.CONFIDENCE_THRESHOLD:
            return ABSTENTION, [], None

        extracted = self._extract_best_passage(question, top_doc["content"])
        doc_id = top_doc.get("doc_id", "")
        title = top_doc.get("title", "")
        source_label = f"{title} ({doc_id})" if doc_id else title

        return extracted, [source_label], doc_id


def print_welcome():
    print(
        "Welcome to the Solstice Learning Q&A assistant.\n"
        "I can answer questions covered by these topics:\n"
        "- Password resets\n"
        "- Inviting team members\n"
        "- CSV report exports\n"
        "- Billing cycles\n"
        "- Single sign-on (SSO)\n"
        "- API rate limits\n"
        "Answers are based only on the documentation and include source citations. "
        "If the docs don't cover a question, I'll say so."
    )


def run_chat(agent: DocsQAAgent):
    print('Ask a question, or type "quit" to exit.')
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.casefold() in {"quit", "exit", "q"}:
            break
        if not question:
            print("Please enter a question.")
            continue

        answer, sources, _ = agent.answer_question(question)
        print(f"\n{answer}")
        if sources:
            print("\nSources:")
            for source in sources:
                print(f"- {source}")


def run_evaluations(agent: DocsQAAgent) -> int:
    q_file = find_questions_file()
    if not q_file:
        print("Warning: example_questions.json not found. Skipping automated suite.")
        return 0

    with q_file.open(encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "example_questions" in raw:
        questions = raw["example_questions"]
    elif isinstance(raw, list):
        questions = raw
    else:
        questions = list(raw.values())

    print("=" * 80)
    print("RUNNING BENCHMARK EVALUATIONS")
    print("=" * 80)

    failures = 0
    for ex in questions:
        q_text = ex["question"]
        expected = ex.get("expected_doc_id")

        answer, sources, cited_id = agent.answer_question(q_text)
        passed = (cited_id == expected)
        status = "PASS" if passed else "FAIL"
        failures += not passed

        print(f"\n[{status}] {q_text}")
        print(answer)
        if sources:
            print("Sources: " + "; ".join(sources))
        print(f"Expected document : {expected or 'no supporting document'}")
        print(f"Cited document IDs: {cited_id or 'none'}")

    print(f"\nEvaluation summary: {len(questions) - failures}/{len(questions)} passed.")
    print("=" * 80)
    return failures


def main():
    parser = argparse.ArgumentParser(description="Solstice Learning Docs Assistant (CLI)")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmark suite only and exit.")
    args = parser.parse_args()

    agent = DocsQAAgent()

    if args.evaluate:
        sys.exit(int(run_evaluations(agent) > 0))

    run_evaluations(agent)
    print_welcome()
    run_chat(agent)


if __name__ == "__main__":
    main()
