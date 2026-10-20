# Architecture \& Solution Design: Solstice Docs Q\&A Assistant

## 1\. Executive Summary

The Solstice Learning Docs Q\&A Assistant is a lightweight, zero-dependency information retrieval and grounded question-answering agent. Designed to operate completely offline without external APIs, vector databases, or deep learning models, the system delivers high-precision answers derived exclusively from internal documentation. A central design goal is **zero hallucination tolerance**: out-of-domain questions and queries regarding unsupported features fail gracefully with an explicit abstention notice.

\---

## 2\. Key Architecture Components

```
                   +---------------------------------------------+
                   |                User Interface               |
                   |   - CLI Chat / Eval Runner (main.py)        |
                   |   - Single-Page Web App (app.py :8080)       |
                   +---------------------------------------------+
                                          |
                                    User Query
                                          v
+-------------------------------------------------------------------------------+
| DocsQAAgent Pipeline                                                          |
|                                                                               |
|  1. Preprocessing \& Normalization                                             |
|     - Tokenize \& suffix-strip                                                 |
|     - Filter STOPWORDS \& CONVERSATIONAL\_WORDS                                 |
|                                                                               |
|  2. Out-of-Domain Guardrail                                                   |
|     - Content tokens checked against `global\_vocab`                           |
|     - Unknown entity detected? ---------> \[RETURN ABSTENTION]                 |
|                                                                               |
|  3. BM25 Retrieval \& Title Weighting                                          |
|     - Calculate BM25 scores (k1=1.5, b=0.75, title\_boost=3.0)                |
|     - Max score < CONFIDENCE\_THRESHOLD (1.6)? -> \[RETURN ABSTENTION]          |
|                                                                               |
|  4. Sentence Overlap Extractive Summarizer                                     |
|     - Split winning document into sentences                                   |
|     - Score by query token density: overlap / sqrt(len(sentence))             |
|     - Preserve document flow order for top-scoring sentences                  |
+-------------------------------------------------------------------------------+
                                          |
                                Extracted Passage + Citation
                                          v
                                    User Response
```

\---

## 3\. Detailed Component Breakdown

### 3.1 Document Ingestion \& Normalization (`load\_documents`)

* **Multi-Source Support**: Recursively resolves knowledge files from Markdown directories (`docs/\*.md`) or structured JSON dumps (`docs.json`, `data/docs.json`).
* **Metadata Extraction**: Regular expressions extract explicit Document IDs (e.g., `Document ID: D-01`) and Markdown headings (`# Title`) while cleaning content bodies.
* **Rule-Based Stemmer (`normalize\_term`)**: A deterministic suffix stripper handles standard English plurals and inflectional morphology (`-ing`, `-es`, `-s`, `-ed`, `-ly`, `-tion`, `-ment`) without heavy NLP dependencies like NLTK or SpaCy.

### 3.2 Probabilistic Ranking (`BM25Retriever`)

BM25 (Best Matching 25) calculates query-document relevance:

$$\\text{IDF}(q\_t) = \\ln \\left(1 + \\frac{N - n(q\_t) + 0.5}{n(q\_t) + 0.5}\\right)$$

$$\\text{Score}(D, Q) = \\sum\_{q\_t \\in Q} \\text{IDF}(q\_t) \\cdot \\frac{f(q\_t, D) \\cdot (k\_1 + 1)}{f(q\_t, D) + k\_1 \\cdot \\left(1 - b + b \\cdot \\frac{|D|}{\\text{avgdl}}\\right)}$$

* **Hyperparameters**:

  * $k\_1 = 1.5$: Calibrates term frequency saturation.
  * $b = 0.75$: Penalizes long documents.
  * $\\text{title\_boost} = 3.0$: Emphasizes matching title terms.

### 3.3 Strict Abstention \& Anti-Hallucination Framework

The system uses a 3-layer guardrail to avoid giving ungrounded answers:

1. **Conversational Term Pruning**: Isolates substantive subject matter tokens by stripping auxiliary/conversational words (`help`, `guide`, `trouble`, `can`, `want`) alongside standard stopwords.
2. **Global Vocabulary Filter**: If an isolated subject token does not exist anywhere within the corpus vocabulary (e.g., *"Salesforce"*, *"HubSpot"*, *"Zapier"*), the query is rejected immediately without scoring.
3. **Relevance Thresholding**: Queries with marginal token overlap that fail to reach `CONFIDENCE\_THRESHOLD = 1.6` are abstained:

> \*"I don't have an answer for that in the documentation."\*

### 3.4 Extractive Passage Scoring (`\_extract\_best\_passage`)

Instead of returning whole documents, the agent extracts focused answers:

* The matched document is split into individual sentences using sentence-boundary lookbehind regexes (`(?<=\[.!?])\\s+`).
* Sentences are scored based on query token overlap penalized by sentence length:

$$\\text{Sentence Score} = \\frac{|T\_{\\text{sentence}} \\cap T\_{\\text{query}}|}{\\sqrt{|T\_{\\text{sentence}}|}}$$

* The top $N$ sentences (default: 2) are reassembled in their original document order to preserve narrative cohesion.

\---

## 4\. Evaluation and Verification

The system includes a verification harness (`main.py --evaluate`) running against `example\_questions.json`:

|ID|Test Question|Category|Expected Target|Agent Result|Status|
|-|-|-|-|-|-|
|**Q-1**|*How do I reset my password?*|Retrieval|`D-01`|`D-01`|**PASS**|
|**Q-2**|*How long do I have to use a password reset link before it expires?*|Retrieval / Extraction|`D-01`|`D-01`|**PASS**|
|**Q-3**|*What's the maximum number of rows I can export to CSV?*|Numeric Detail|`D-03`|`D-03`|**PASS**|
|**Q-4**|*What are the API rate limits for the Growth plan?*|Tier Query|`D-06`|`D-06`|**PASS**|
|**Q-5**|*Can I connect Solstice to Salesforce?*|Grounded Abstention|`None`|`None` (Abstain)|**PASS**|

**Benchmark Score**: 5 / 5 (100% Precision \& Abstention Reliability).

\---

