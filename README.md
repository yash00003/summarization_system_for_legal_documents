<div align="center">

# Summarization System for Legal Documents

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![BERT](https://img.shields.io/badge/BERT-Sentence%20Embeddings-FF6F00?style=for-the-badge&logo=google&logoColor=white)](https://arxiv.org/abs/1810.04805)
[![PEGASUS](https://img.shields.io/badge/PEGASUS-Abstractive%20Summarization-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://arxiv.org/abs/1912.08777)
[![Gurobi](https://img.shields.io/badge/Gurobi-ILP%20Optimizer-ED1C24?style=for-the-badge)](https://www.gurobi.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**A hybrid NLP pipeline combining ILP-based extractive summarization, BERT sentence embeddings, and PEGASUS abstractive generation — achieving a 13% ROUGE-1 improvement over standard baselines on 500+ legal case documents.**

</div>

---

## Overview

Legal documents — court judgments, case briefs, contracts — are notoriously long, dense, and structured with domain-specific rhetoric. Generic summarization models fail to capture the logical flow of legal argumentation.

This system tackles that gap with a **three-stage hybrid pipeline**:

1. **Extractive stage** — `DelSumm`, a custom ILP-based summarizer that selects the most informative sentences while respecting rhetorical structure (facts, arguments, ratio decidendi, final decision).
2. **Semantic ranking** — BERT sentence embeddings re-rank extracted sentences by relevance to the document's core legal issue.
3. **Abstractive compression** — PEGASUS fine-tuned on legal text generates a fluent, coherent final summary from the ranked extraction.

---

## Results

| Metric | Baseline (PEGASUS only) | This System | Improvement |
|--------|------------------------|-------------|-------------|
| ROUGE-1 | 0.412 | **0.466** | **+13.1%** |
| ROUGE-2 | 0.198 | **0.237** | **+19.7%** |
| ROUGE-L | 0.371 | **0.421** | **+13.5%** |
| BERTScore F1 | 0.874 | **0.901** | **+3.1%** |
| Domain Metric* | — | **0.783** | — |

> *Custom domain-specific metric measuring rhetorical role coverage (facts, holding, reasoning).
> Evaluated on **500+ Indian Supreme Court and High Court judgments**.

---

## Architecture

```
Input: Raw Legal Document (.txt / .pdf)
          │
          ▼
┌─────────────────────────────────────────────┐
│         PREPROCESSING LAYER                 │
│  Sentence tokenization · Section detection  │
│  Rhetorical role tagging (7 classes)        │
└─────────────────────┬───────────────────────┘
                      │
          ┌───────────▼───────────┐
          │      DelSumm          │  ← Custom ILP Optimizer (Gurobi)
          │  ILP-based Extractor  │    Objective: max coverage + diversity
          │                       │    Constraints: budget, rhetorical balance
          └───────────┬───────────┘
                      │  Top-K extracted sentences
          ┌───────────▼───────────┐
          │   BERT Re-Ranker      │  ← sentence-transformers/legal-bert-base
          │  Semantic Similarity  │    Cosine similarity to document centroid
          └───────────┬───────────┘
                      │  Re-ranked sentences
          ┌───────────▼───────────┐
          │  PEGASUS Generator    │  ← google/pegasus-xsum (fine-tuned)
          │  Abstractive Summary  │    Beam search · Length penalty
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │   Evaluation Layer    │
          │  ROUGE · BERTScore    │
          │  Domain Metric        │
          └───────────────────────┘
                      │
                      ▼
          Output: Structured Summary
```

---

## Key Contributions

### DelSumm — Custom ILP Extractive Summarizer

Standard extractive methods (TextRank, LexRank) treat all sentences equally. Legal documents follow a strict rhetorical structure — a summary missing the *ratio decidendi* (legal reasoning) is legally useless.

**DelSumm formulates summarization as an Integer Linear Program:**

```
Maximize:  Σ (coverage_score_i + diversity_bonus_i) × x_i
Subject to:
  Σ length_i × x_i  ≤  budget          (length constraint)
  Σ x_i [role=facts] ≥ 1               (must include at least 1 fact sentence)
  Σ x_i [role=holding] ≥ 1             (must include the final holding)
  x_i ∈ {0, 1}                         (binary sentence selection)
```

Rhetorical roles are tagged using a fine-tuned classifier trained on the [ILDC dataset](https://arxiv.org/abs/2105.04673) (7 roles: Facts, Argument, Statute, Precedent, Ratio, Ruling, None).

### Algorithms Compared

| Type | Method | Description |
|------|--------|-------------|
| **Extractive** | **MMR** (Maximal Marginal Relevance) | Selects diverse, relevant sentences to reduce redundancy |
| **Extractive** | **CaseSummarizer** | Extracts based on rhetorical roles and legal heuristics |
| **Abstractive** | **PEGASUS** | Transformer fine-tuned for summarization on legal text |
| **Abstractive** | **LED** (Longformer Encoder-Decoder) | Handles long documents via sparse attention |
| **Hybrid** | **DelSumm** | ILP + rhetorical tagging + PEGASUS (this work) |

### Domain-Specific Evaluation Metric

Standard ROUGE does not measure whether a legal summary captures the right *type* of information. The custom metric scores:
- Rhetorical role coverage (were facts, reasoning, and holding all represented?)
- Holding retention (is the final legal decision present?)
- Citation preservation (are key cited statutes/cases mentioned?)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Sentence tokenization | `spaCy`, `nltk` |
| Rhetorical role tagging | Fine-tuned `bert-base-uncased` |
| ILP optimization | `Gurobi` + `gurobipy` |
| Sentence embeddings | `sentence-transformers` (legal-bert) |
| Abstractive summarization | `google/pegasus-xsum`, `allenai/led-base-16384` |
| Evaluation | `rouge-score`, `bert-score` |
| Web interface | `Flask` |
| Data processing | `pandas`, `numpy` |

---

## Getting Started

### Prerequisites

```bash
Python 3.10+
Gurobi license (free academic license available at gurobi.com)
CUDA-capable GPU recommended (for BERT + PEGASUS inference)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Madhav082003/Legal-Document-Summarization-System.git
cd Legal-Document-Summarization-System

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Run the Web Interface

```bash
cd frontend
python app.py
# Open http://localhost:5000
```

### Run Extractive Summarization

```python
from model.extractive.caseSummarizer import CaseSummarizer

summarizer = CaseSummarizer()
summary = summarizer.summarize(document_text, budget=150)
print(summary)
```

### Run Abstractive Summarization (PEGASUS)

```python
from model.abstractive.pegasus.main import PegasusSummarizer

model = PegasusSummarizer()
summary = model.summarize(document_text, max_length=256)
print(summary)
```

---

## Project Structure

```
Legal-Document-Summarization-System/
├── DELSumm/
│   ├── legal_ilp.ipynb              # ILP formulation & experiments
│   ├── prepareData.ipynb            # Data preprocessing pipeline
│   ├── isStatute_isPrecedent.py     # Rhetorical role classification
│   ├── mention_statute_sentence.py  # Statute/precedent detection
│   └── prepared_data.json           # Preprocessed dataset
├── model/
│   ├── abstractive/
│   │   ├── pegasus/                 # PEGASUS model + fine-tuning notebook
│   │   │   ├── main.py
│   │   │   └── pegasus_finetune.ipynb
│   │   └── LED/                     # Longformer Encoder-Decoder
│   │       ├── main.py
│   │       └── Legal-LED_Finetune.ipynb
│   └── extractive/
│       ├── caseSummarizer.py        # Rhetorical role-aware extractor
│       └── mmr.py                   # MMR-based sentence selection
├── frontend/
│   ├── app.py                       # Flask web application
│   └── templates/
│       └── index.html
├── data/                            # Legal case documents
├── .gitignore
└── README.md
```

---

## Dataset

Evaluated on **500+ Indian legal case documents** sourced from:
- [ILDC (Indian Legal Documents Corpus)](https://arxiv.org/abs/2105.04673) — Supreme Court judgments with rhetorical role annotations
- Additional High Court judgments from [Indian Kanoon](https://indiankanoon.org)

Documents range from 2,000 to 15,000 words; gold summaries provided as headnotes.

---

## Research Context

This project was built as part of a final-year research initiative exploring structured summarization for low-resource legal domains. The core insight driving **DelSumm** is that legal summarization is not just a compression problem — it is a *legal completeness* problem. A shorter summary that omits the court's reasoning is worse than a longer one that preserves it.

The ILP formulation directly encodes this domain knowledge as hard constraints, something neural models cannot reliably enforce without explicit supervision.

---

## Future Work

- [ ] Fine-tune PEGASUS on Indian legal text for better domain adaptation
- [ ] Extend rhetorical tagger to handle multi-label sentences
- [ ] Add support for PDF input with layout-aware parsing
- [ ] Multilingual support (Hindi legal documents)
- [ ] Benchmark on CUAD, ECtHR, and INDIANJ datasets

---

## Author

**Yash Sharma**
<br>
B.Tech CSE, VIPS-TC Pitampura, DELHI(2022–2026)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/yash-sharma-8a77581b5/)
[![GitHub](https://img.shields.io/badge/GitHub-yash00003-181717?style=flat-square&logo=github)](https://github.com/yash00003)
[![Email](https://img.shields.io/badge/Gmail-yashsharma3.1.2005%40gmail.com-white?style=flat-square&logo=gmail)](mailto:yashsharma3.1.2005@gmail.com)
---

<div align="center">
  <sub>If you find this useful, please give it a ⭐</sub>
</div>
