import json
import math
import os
import re
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR / "docs"
DATA_DIR = APP_DIR / "data"
STARTER_DATA_DIR = APP_DIR / "starter_kit" / "data"

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

    def answer_question(self, question: str) -> Dict[str, Any]:
        raw_words = re.findall(r"\b[a-z0-9]+\b", question.lower())
        if not raw_words:
            return {"answer": ABSTENTION, "sources": [], "matched": False, "score": 0.0}

        norm_tokens = [normalize_term(w) for w in raw_words if w not in STOPWORDS]
        test_tokens = [t for t in norm_tokens if t not in CONVERSATIONAL_WORDS]

        if test_tokens:
            unknown_entities = [t for t in test_tokens if t not in self.retriever.global_vocab]
            if unknown_entities:
                return {"answer": ABSTENTION, "sources": [], "matched": False, "score": 0.0}

        results = self.retriever.score_query(question)
        if not results:
            return {"answer": ABSTENTION, "sources": [], "matched": False, "score": 0.0}

        top_score, top_doc = results[0]

        if top_score < self.CONFIDENCE_THRESHOLD:
            return {"answer": ABSTENTION, "sources": [], "matched": False, "score": round(top_score, 2)}

        extracted = self._extract_best_passage(question, top_doc["content"])
        doc_id = top_doc.get("doc_id", "")
        title = top_doc.get("title", "")
        source_label = f"{title} ({doc_id})" if doc_id else title

        return {
            "answer": extracted,
            "sources": [source_label],
            "matched": True,
            "score": round(top_score, 2)
        }


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solstice Learning Q&A Assistant</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f1f5f9;
            margin: 0;
            padding: 30px 15px;
            display: flex;
            justify-content: center;
        }
        .container {
            width: 100%;
            max-width: 780px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            padding: 32px;
        }
        h1 { margin: 0 0 6px; font-size: 24px; color: #0f172a; }
        p.subtitle { margin: 0 0 20px; color: #64748b; font-size: 14px; }
        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 22px;
        }
        .chip {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 12px;
            cursor: pointer;
            color: #334155;
            transition: all 0.2s;
        }
        .chip:hover { background: #e2e8f0; }
        .input-box {
            display: flex;
            gap: 10px;
            margin-bottom: 24px;
        }
        input[type="text"] {
            flex: 1;
            padding: 13px 16px;
            border: 1.5px solid #cbd5e1;
            border-radius: 8px;
            font-size: 15px;
            outline: none;
        }
        input[type="text"]:focus { border-color: #2563eb; }
        button {
            padding: 13px 24px;
            background: #2563eb;
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        button:hover { background: #1d4ed8; }
        .card {
            display: none;
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
            line-height: 1.55;
        }
        .card.success {
            background: #f0fdf4;
            border-left: 5px solid #22c55e;
            color: #14532d;
        }
        .card.abstain {
            background: #fffbeb;
            border-left: 5px solid #f59e0b;
            color: #78350f;
        }
        .meta {
            margin-top: 10px;
            font-size: 13px;
            font-weight: 600;
            color: #475569;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Solstice Learning - Docs Assistant</h1>
        <p class="subtitle">Search product manuals with strict hallucination-free domain grounding.</p>
        
        <div class="chips">
            <span class="chip" onclick="ask('How to reset password')">Password Reset</span>
            <span class="chip" onclick="ask('Inviting team members')">Inviting Team Members</span>
            <span class="chip" onclick="ask('Exporting Reports to CSV')">CSV Export</span>
            <span class="chip" onclick="ask('Understanding Billing Cycles')">Billing Cycles</span>
            <span class="chip" onclick="ask('API Rate Limits')">API Limits</span>
            <span class="chip" onclick="ask('Can I connect Solstice to Salesforce?')">Salesforce</span>
        </div>

        <form class="input-box" onsubmit="handleQuery(event)">
            <input type="text" id="qInput" placeholder="Type a question..." required autofocus />
            <button type="submit">Search</button>
        </form>

        <div id="outputCard" class="card">
            <div id="ansText"></div>
            <div id="metaText" class="meta"></div>
        </div>
    </div>

    <script>
        function ask(text) {
            document.getElementById('qInput').value = text;
            handleQuery(new Event('submit'));
        }

        async function handleQuery(e) {
            e.preventDefault();
            const q = document.getElementById('qInput').value.trim();
            if (!q) return;

            const card = document.getElementById('outputCard');
            const ans = document.getElementById('ansText');
            const meta = document.getElementById('metaText');

            ans.innerText = "Searching documentation...";
            meta.innerText = "";
            card.style.display = "block";
            card.className = "card";

            try {
                const res = await fetch('/api/ask?q=' + encodeURIComponent(q));
                const data = await res.json();
                ans.innerText = data.answer;

                if (data.matched) {
                    card.className = "card success";
                    meta.innerText = "Source: " + data.sources.join(', ') + " | Score: " + data.score;
                } else {
                    card.className = "card abstain";
                    meta.innerText = "Grounded Abstention: No matching subject in documentation.";
                }
            } catch (err) {
                card.className = "card abstain";
                ans.innerText = "Error querying local documentation service.";
            }
        }
    </script>
</body>
</html>
"""

agent_instance = DocsQAAgent()


class WebServerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/ask":
            params = parse_qs(parsed.query)
            q = params.get("q", [""])[0]
            result = agent_instance.answer_question(q)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))

    def log_message(self, format, *args):
        return


def main():
    port = 8000
    server = HTTPServer(("127.0.0.1", port), WebServerHandler)
    print("=" * 65)
    print(f" Web app running at: http://127.0.0.1:{port}")
    print(" Press Ctrl+C to terminate the server.")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
