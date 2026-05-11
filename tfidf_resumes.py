"""
Stage 3 — TF-IDF Computation (Standard Library Only)
-----------------------------------------------------
Uses normalized skill lists from Stage 2 to compute TF-IDF scores
for all 10 resumes using manual formulas (no numpy/pandas/sklearn).

Formulas:
  TF(skill, resume) = 1 / N           where N = # unique skills in resume
  IDF(skill)        = ln(10 / df)     where df = # resumes containing skill
  TF-IDF            = TF × IDF
"""

import math
from skill_normalizer import normalize_skills

# ── Resume data (same as Stage 2) ──────────────────────────────────────────
RAW_RESUMES = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

TOTAL_DOCS = len(RAW_RESUMES)  # 10


def main():
    # ── Normalize all resumes (suppress intermediate prints) ────────────
    # Temporarily redirect prints from normalize_skills
    import io, sys
    resumes = []
    for name, raw in RAW_RESUMES:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()       # swallow normalize_skills prints
        skills = normalize_skills(raw)
        sys.stdout = old_stdout
        resumes.append((name, skills))

    # ════════════════════════════════════════════════════════════════════
    # STEP 1 — Shared vocabulary
    # ════════════════════════════════════════════════════════════════════
    vocab_set = set()
    for _, skills in resumes:
        vocab_set.update(skills)
    vocabulary = sorted(vocab_set)

    sep = "═" * 80
    print(sep)
    print("  STEP 1: SHARED VOCABULARY")
    print(sep)
    print(f"  Total unique canonical skills: {len(vocabulary)}\n")
    for i, skill in enumerate(vocabulary):
        print(f"    [{i:2d}] {skill}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 2 — Compute document frequency (df) for each skill
    # ════════════════════════════════════════════════════════════════════
    df = {}  # skill → number of resumes containing it
    for skill in vocabulary:
        count = sum(1 for _, skills in resumes if skill in skills)
        df[skill] = count

    print(f"\n{sep}")
    print("  STEP 2: DOCUMENT FREQUENCY (df) & IDF")
    print(sep)
    print(f"  {'Skill':<35s} {'df':>4s}   {'IDF = ln(10/df)':>15s}")
    print(f"  {'─' * 35} {'─' * 4}   {'─' * 15}")
    for skill in vocabulary:
        idf_val = math.log(TOTAL_DOCS / df[skill])
        print(f"  {skill:<35s} {df[skill]:>4d}   {idf_val:>15.6f}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 3 — Per-resume TF-IDF table
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  STEP 3: TF-IDF SCORES PER RESUME")
    print(sep)

    # Store results for optional downstream use
    tfidf_vectors = {}  # name → {skill: tfidf, ...}

    for idx, (name, skills) in enumerate(resumes, start=1):
        n = len(skills)
        tf_val = 1.0 / n  # uniform TF for this resume

        print(f"\n  ┌─ [{idx:02d}] {name}  (N = {n}, TF = 1/{n} = {tf_val:.6f})")
        print(f"  │  {'Skill':<35s} {'TF':>10s}  {'IDF':>10s}  {'TF-IDF':>10s}")
        print(f"  │  {'─' * 35} {'─' * 10}  {'─' * 10}  {'─' * 10}")

        resume_tfidf = {}
        for skill in sorted(skills):
            idf_val = math.log(TOTAL_DOCS / df[skill])
            tfidf = tf_val * idf_val
            resume_tfidf[skill] = tfidf
            print(f"  │  {skill:<35s} {tf_val:>10.6f}  {idf_val:>10.6f}  {tfidf:>10.6f}")

        tfidf_vectors[name] = resume_tfidf
        print(f"  └{'─' * 72}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 4 — Full TF-IDF matrix (sparse view)
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  STEP 4: FULL TF-IDF MATRIX (top skills per candidate)")
    print(sep)

    for name, scores in tfidf_vectors.items():
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = ", ".join(f"{s} ({v:.4f})" for s, v in ranked[:3])
        print(f"  {name:<20s} │ Top-3: {top3}")

    # ════════════════════════════════════════════════════════════════════
    # STATS — Quick sanity checks
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  SANITY CHECKS")
    print(sep)

    # Skills appearing in only 1 resume have highest IDF = ln(10/1) = ln(10)
    max_idf = math.log(TOTAL_DOCS / 1)
    rare_skills = [s for s in vocabulary if df[s] == 1]
    print(f"  Max possible IDF: ln(10) = {max_idf:.6f}")
    print(f"  Skills with df=1 (rarest, {len(rare_skills)} total):")
    for s in rare_skills:
        print(f"    • {s}")

    # 'python' appears most, should have lowest IDF
    python_idf = math.log(TOTAL_DOCS / df["python"])
    print(f"\n  'python' df={df['python']}, IDF = ln(10/{df['python']}) = {python_idf:.6f} (lowest)")
    print(sep)


if __name__ == "__main__":
    main()
