# Solstice Learning - Docs Q\&A Assistant

A zero-dependency, grounded question-answering system built strictly with the Python standard library. The assistant retrieves relevant documentation, extracts precise answering sentences, and actively abstains from answering out-of-domain questions to prevent hallucinations.

\---

## Features

* **Zero External Dependencies**: Implemented purely using Python's standard libraries (`re`, `math`, `http.server`, `urllib`, `pathlib`, `json`).
* **Custom BM25 Ranking Engine**: Probabilistic document retrieval with custom sublinear term-frequency scaling, document length normalization, and configurable title boosting.
* **Strict Hallucination Prevention \& Abstention**:

  * Global vocabulary containment checks filter out unseen named entities (e.g., third-party tools like *Salesforce*).
  * Stopword and conversational-filler filtering isolate actual domain intent.
  * BM25 score thresholding abstains with standard messaging when relevance confidence is insufficient.
* **Sentence-Level Passage Extraction**: Evaluates sentence overlap using token-density heuristics to deliver concise, contextual answers instead of dumping entire documents.
* **Dual Interface Modes**:

  * **CLI / Evaluation Mode (`main.py`)**: Runs built-in test suites and interactive terminal chat sessions.
  * **Web GUI Mode (`app.py`)**: Single-page modern web client with preset quick-test chips, dynamic cards, and source attribution badges.

\---

## Project Structure

```text
.
├── data/

│   ├── docs.json

│   └── example\_questions.json

├── main.py

├── README.md

└── SOLUTION\_DESIGN.md```

\---

## Getting Started

### Prerequisites

* **Python 3.8+** (No `pip install` required)

### 1\. Run Automated Evaluations (CLI)

Run the test suite against `example\_questions.json` to verify grounding and abstention accuracy:

```bash
python3 main.py --evaluate
```

Expected output:

```text
================================================================================
RUNNING BENCHMARK EVALUATIONS
================================================================================

\[PASS] How do I reset my password?
...
\[PASS] Can I connect Solstice to Salesforce?
I don't have an answer for that in the documentation.
Expected document : no supporting document
Cited document IDs: none

Evaluation summary: 5/5 passed.
================================================================================
```

### 2\. Interactive Terminal Mode

Start an interactive conversation in the terminal:

```bash
python3 main.py
```

Type any documentation query or type `quit` / `exit` to end the session.

### 3\. Launch Web UI

Run the local web application:

```bash
python3 app.py
```

Open your browser at [**http://127.0.0.1:8080**](http://127.0.0.1:8080).

\---

## How It Works

1. **Ingestion**: Documents are loaded either from `.md` files in `docs/` or directly from `docs.json`. Document IDs and headers are parsed and indexed.
2. **Indexing**: Documents are tokenized with rule-based stemming (suffixes like `-ing`, `-es`, `-ly`, `-ment`). Term frequencies and inverse document frequencies (IDF) are precomputed.
3. **Guardrail Check**:

   * The query is stripped of standard English stopwords and common conversational fillers (`"can i"`, `"how do i"`, `"help"`, `"tell me"`).
   * If the query contains content terms absent from the global corpus vocabulary, the system immediately returns:

> \*"I don't have an answer for that in the documentation."\*

4. **Retrieval**: Queries passing the guardrail are scored with the BM25 formula against title and body text.
5. **Score Gating**: If the top document score falls below `1.6`, the request is abstained.
6. **Passage Extraction**: Sentences in the matched document are ranked by normalized query overlap to extract the top 1–2 most relevant sentences alongside source citations.

