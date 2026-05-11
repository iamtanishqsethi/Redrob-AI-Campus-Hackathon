# 🎯 Resume–JD Skill Matching Pipeline

A **pure-Python** (zero dependencies) pipeline that normalizes messy resume/JD skill strings into canonical tokens, computes TF-IDF vectors, and ranks candidates against job descriptions using cosine similarity.

Built for the **Redrob AI Campus Hackathon**.

---

## 📁 Project Structure

```
Redrob AI Campus Hackathon/
├── main.py                # ★ Single entry point — runs full pipeline & prints submission answers
├── skill_normalizer.py    # Stage 1 — Core normalization engine
├── process_resumes.py     # Stage 2 — Batch-process 10 resumes
├── tfidf_resumes.py       # Stage 3 — TF-IDF computation
├── jd_vectors.py          # Stage 4 — JD binary vector construction
├── cosine_match.py        # Stage 5 — Cosine similarity ranking
├── sanity_checks.py       # Stage 6 — End-to-end validation (40 checks)
└── README.md
```

---

## 🧩 Pipeline Stages

### Stage 1 — Skill Normalization (`skill_normalizer.py`)

The foundation of the pipeline. Takes a raw comma-separated skill string and returns a clean, deduplicated list of canonical skill names.

**Processing steps:**
1. **Lowercase** the entire input string
2. **Protect multi-word phrases** (e.g. `"deep learning"` → `"deep_learning"`) before splitting — handles hyphens, extra spaces, and slashes
3. **Split** on commas, strip whitespace
4. **Clean** each token — collapse hyphens/spaces/slashes to underscores, strip non-alphanumeric characters
5. **Alias map** — look up each token in `SKILL_ALIASES`; discard unknowns
6. **Deduplicate** — preserve insertion order

**Key features:**
- Handles typos: `Pyhton` → `python`, `kubernates` → `kubernetes`, `TypeScrpit` → `typescript`
- Handles format variations: `Node.JS`, `vue.js`, `HTML/CSS`, `CI/CD`, `UI/UX`
- Maps related tools to canonical categories: `matplotlib` → `data_visualization`, `sklearn` → `machine_learning`
- 13 protected multi-word phrases, 140+ alias mappings

---

### Stage 2 — Resume Processing (`process_resumes.py`)

Batch-processes 10 candidate resumes through the normalizer, printing intermediate steps for each.

**Resumes:**

| # | Candidate | Raw Skills |
|---|-----------|-----------|
| 01 | Arjun Sharma | Python, ML, SQL, pandas, numpy, Deep Learning |
| 02 | Priya Nair | JavaScript, React, Node.js, MongoDB, REST API, HTML/CSS |
| 03 | Rahul Gupta | Java, Spring Boot, MySQL, Microservices, Docker, Kubernetes |
| 04 | Sneha Patel | Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib |
| 05 | Vikram Singh | C++, Algorithms, Data Structures, CP, Python |
| 06 | Ananya Krishnan | JavaScript, Vue.js, Python, Flask, PostgreSQL, AWS, CI/CD |
| 07 | Karan Mehta | Python, Sklearn, XGBoost, Feature Engineering, SQL, Tableau |
| 08 | Deepika Rao | Java, Android, Kotlin, Firebase, REST, UI/UX, Figma |
| 09 | Aditya Kumar | React, TypeScript, GraphQL, Redux, Tailwind, Node.js, Jest |
| 10 | Meera Iyer | Python, R, Statistics, ML, Regression, Clustering, Power BI |

---

### Stage 3 — TF-IDF Computation (`tfidf_resumes.py`)

Computes TF-IDF scores using **only the standard library** (`math.log`).

**Formulas:**
```
TF(skill, resume)  = 1 / N              where N = unique skills in resume
IDF(skill)         = ln(10 / df)         where df = resumes containing skill
TF-IDF             = TF × IDF
```

**Output includes:**
- Shared vocabulary (50 skills, alphabetically indexed)
- Document frequency and IDF for every skill
- Per-resume TF-IDF tables with TF, IDF, TF-IDF columns
- Top-3 discriminating skills per candidate

---

### Stage 4 — JD Binary Vectors (`jd_vectors.py`)

Builds binary skill vectors for 3 job descriptions using the **same normalization pipeline**.

| JD | Company / Role | Active Skills |
|---|---|---|
| JD-1 | Kakao — ML Engineer | 10 skills |
| JD-2 | Naver — Backend Engineer | 9 skills |
| JD-3 | Line — Frontend Engineer | 11 skills |

**Output includes:**
- Per-JD normalization trace (intermediate steps)
- Binary vector with Required/Preferred labels
- Side-by-side comparison matrix
- JD overlap analysis

---

### Stage 5 — Cosine Similarity Matching (`cosine_match.py`)

Computes cosine similarity for all 30 resume × JD pairs and ranks candidates.

```
cosine(A, B) = dot(A, B) / (‖A‖ × ‖B‖)
```

Where `A` = resume TF-IDF vector (50-dim), `B` = JD binary vector (50-dim).

**Final Results — Top 3 per JD:**

| JD | #1 | #2 | #3 |
|---|---|---|---|
| Kakao ML Engineer | Sneha Patel (0.59) | Arjun Sharma (0.40) | Karan Mehta (0.40) |
| Naver Backend | Rahul Gupta (0.81) | Ananya Krishnan (0.28) | Deepika Rao (0.19) |
| Line Frontend | Aditya Kumar (0.67) | Priya Nair (0.58) | Ananya Krishnan (0.35) |

---

### Stage 6 — Sanity Checks (`sanity_checks.py`)

Runs **40 automated checks** validating every stage of the pipeline:

| Check | What it validates | Count |
|---|---|---|
| 1. Normalization | Alias mappings, typo handling, deduplication | 9 |
| 2. Vocabulary | Size, union match, sort order, no duplicates | 4 |
| 3. IDF | Highest/lowest values, positivity | 3 |
| 4. Vectors | Component bounds (≤ 1.0), vector lengths | 12 |
| 5. JD integrity | Binary-only, vocab membership, no missing bits | 12 |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (uses `list[str]` type hints)
- **No external packages required** — entire pipeline uses only the standard library

### Verify Python version

```bash
python3 --version
# Python 3.10.x or higher
```

---

## ▶️ Execution — Run Each Stage

Run all commands from the project root directory:

```bash
cd "Redrob AI Campus Hackathon"
```

### Stage 1 — Test the normalizer on a single input

```bash
python3 skill_normalizer.py
```

**Expected output:**
```
Raw input: 'Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning'

  [1] Lowercased   → 'pyhton, machinelearning, sql, pandas, numpy, deep-learning'
  [2] Protected    → 'pyhton, machinelearning, sql, pandas, numpy, deep_learning'
  [3] Split tokens → ['pyhton', 'machinelearning', 'sql', 'pandas', 'numpy', 'deep_learning']
  [4] Cleaned      → ['pyhton', 'machinelearning', 'sql', 'pandas', 'numpy', 'deep_learning']
  [5] Alias-mapped → ['python', 'machine_learning', 'sql', 'pandas', 'numpy', 'deep_learning']
  [6] Deduplicated → ['python', 'machine_learning', 'sql', 'pandas', 'numpy', 'deep_learning']

✅ Final output: ['python', 'machine_learning', 'sql', 'pandas', 'numpy', 'deep_learning']
   ✔ Assertion passed.
```

### Stage 2 — Process all 10 resumes

```bash
python3 process_resumes.py
```

Prints each candidate with intermediate normalization steps, a summary table, and 4 validation checks.

### Stage 3 — Compute TF-IDF

```bash
python3 tfidf_resumes.py
```

Outputs the shared vocabulary, document frequencies, per-resume TF-IDF tables, and top-3 discriminating skills.

### Stage 4 — Build JD binary vectors

```bash
python3 jd_vectors.py
```

Shows per-JD normalization, binary vectors with Required/Preferred labels, side-by-side matrix, and overlap analysis.

### Stage 5 — Rank candidates by cosine similarity

```bash
python3 cosine_match.py
```

Produces full ranked lists (all 10 candidates per JD), top-3 summary, and skill-level match breakdowns.

### Stage 6 — Run all sanity checks

```bash
python3 sanity_checks.py
```

Runs 40 automated checks. Expected final output:

```
  Total checks: 40
  Passed:       40  ✔
  Failed:       0

  🎉 ALL CHECKS PASSED — Pipeline is consistent end-to-end.
```

---

## 🔁 Run Everything at Once (`main.py`)

The easiest way to run the full pipeline and see all submission answers:

```bash
python3 main.py
```

This single command executes all 6 stages and prints the final output:

```
══════════════════════════════════════════════════════════════════════
  REDROB AI CAMPUS HACKATHON — RESUME × JD MATCHING PIPELINE
══════════════════════════════════════════════════════════════════════

▸ STAGE 1 & 2: SKILL NORMALIZATION────────────────────────────────────
  Arjun Sharma         → ['python', 'machine_learning', 'sql', 'pandas', 'numpy', 'deep_learning']
  Priya Nair           → ['javascript', 'react', 'node_js', 'mongodb', 'rest_api', 'html_css']
  Rahul Gupta          → ['java', 'spring_boot', 'mysql', 'microservices', 'docker', 'kubernetes']
  Sneha Patel          → ['python', 'tensorflow', 'keras', 'natural_language_processing', 'bert', 'data_visualization']
  Vikram Singh         → ['cpp', 'algorithms', 'data_structures', 'competitive_programming', 'python']
  Ananya Krishnan      → ['javascript', 'vue', 'python', 'flask', 'postgresql', 'aws', 'ci_cd']
  Karan Mehta          → ['python', 'machine_learning', 'xgboost', 'feature_engineering', 'sql', 'tableau']
  Deepika Rao          → ['java', 'android', 'kotlin', 'firebase', 'rest_api', 'ui_ux', 'figma']
  Aditya Kumar         → ['react', 'typescript', 'graphql', 'redux', 'tailwind', 'node_js', 'jest']
  Meera Iyer           → ['python', 'r', 'statistics', 'machine_learning', 'regression', 'clustering', 'power_bi']

▸ STAGE 3: TF-IDF COMPUTATION─────────────────────────────────────────
  Shared vocabulary: 50 unique canonical skills
  Lowest  IDF: "python" (df=6, IDF=0.5108)
  Highest IDF: ln(10) = 2.3026  (42 skills with df=1)

▸ STAGE 4: JD BINARY VECTORS──────────────────────────────────────────
  JD-1 — Kakao (ML Engineer): 10 active skills
  JD-2 — Naver (Backend Engineer): 9 active skills
  JD-3 — Line (Frontend Engineer): 11 active skills

▸ STAGE 5: COSINE SIMILARITY RANKINGS─────────────────────────────────
  JD-1: Sneha Patel (0.59), Arjun Sharma (0.40), Karan Mehta (0.40) ...
  JD-2: Rahul Gupta (0.81), Ananya Krishnan (0.28), Deepika Rao (0.19) ...
  JD-3: Aditya Kumar (0.67), Priya Nair (0.58), Ananya Krishnan (0.35) ...

▸ STAGE 6: SANITY CHECKS──────────────────────────────────────────────
  ✔ 9/9 checks passed

══════════════════════════════════════════════════════════════════════
  ★  FINAL SUBMISSION ANSWERS  ★
══════════════════════════════════════════════════════════════════════

  JD-1 — Kakao (ML Engineer)
  Sneha Patel(0.59), Arjun Sharma(0.40), Karan Mehta(0.40)

  JD-2 — Naver (Backend Engineer)
  Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

  JD-3 — Line (Frontend Engineer)
  Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

### Run stages individually

```bash
python3 skill_normalizer.py    # Stage 1 — test normalizer
python3 process_resumes.py     # Stage 2 — all 10 resumes
python3 tfidf_resumes.py       # Stage 3 — TF-IDF tables
python3 jd_vectors.py          # Stage 4 — JD vectors
python3 cosine_match.py        # Stage 5 — full rankings
python3 sanity_checks.py       # Stage 6 — 40 validation checks
```

---

## 🧪 Testing Guide

### Unit-level testing

Test the normalizer with custom inputs directly in Python:

```python
from skill_normalizer import normalize_skills

# Test typo handling
assert normalize_skills("Pyhton") == ["python"]

# Test multi-word phrase protection
assert normalize_skills("Deep Learning, Machine Learning") == ["deep_learning", "machine_learning"]

# Test deduplication
assert normalize_skills("Python, python, PYTHON") == ["python"]

# Test slash handling
assert normalize_skills("HTML/CSS, UI/UX") == ["html_css", "ui_ux"]

# Test unknown skill discard
assert normalize_skills("Python, MadeUpSkill123") == ["python"]
```

### Adding a new resume

1. Add the entry to the `RAW_RESUMES` list in `process_resumes.py`, `tfidf_resumes.py`, `cosine_match.py`, and `sanity_checks.py`
2. Update `TOTAL_DOCS` if used
3. If the resume contains new skill terms, add them to `SKILL_ALIASES` in `skill_normalizer.py`
4. Re-run `sanity_checks.py` to verify consistency

### Adding a new JD

1. Add the entry to `JOB_DESCRIPTIONS` in `jd_vectors.py`, `cosine_match.py`, and `sanity_checks.py`
2. If the JD contains new skill terms not in existing resumes, they'll be correctly excluded from the binary vector (since they're outside the shared vocabulary)
3. To include them, add those skills to at least one resume or extend the vocabulary manually

### Extending the alias map

Edit `SKILL_ALIASES` in `skill_normalizer.py`:

```python
SKILL_ALIASES = {
    # Add new mappings here
    "newtypo":       "canonical_name",
    "canonical_name": "canonical_name",
    ...
}
```

If the new skill is a multi-word phrase (e.g. `"natural language processing"`), also add it to the `MULTI_WORD_PHRASES` list.

---

## 📐 Architecture

```
┌──────────────────────┐
│   Raw skill string   │  "Pyhton, MachineLearning, Deep-learning"
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  skill_normalizer.py │  lowercase → protect phrases → split → clean → alias → dedup
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Canonical skill list│  ['python', 'machine_learning', 'deep_learning']
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Resumes │ │   JDs   │
│ TF-IDF  │ │ Binary  │
│ Vectors │ │ Vectors │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           │
           ▼
┌──────────────────────┐
│  Cosine Similarity   │  cosine(TF-IDF, Binary) → ranked candidates
└──────────────────────┘
```

---

## 📜 License

This project was built for the Redrob AI Campus Hackathon.
