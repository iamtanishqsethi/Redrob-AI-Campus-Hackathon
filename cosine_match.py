"""
Stage 5 — Cosine Similarity: Resume × JD Matching
---------------------------------------------------
Builds full-length TF-IDF vectors for resumes and binary vectors for JDs
over the shared 50-skill vocabulary, then ranks candidates per JD using
cosine similarity.  Standard library only.
"""

import io
import math
import sys
from skill_normalizer import normalize_skills

# ── Data ────────────────────────────────────────────────────────────────────
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

JOB_DESCRIPTIONS = {
    "JD-1 — Kakao (ML Engineer)": {
        "required": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
        "preferred": "NLP, BERT, Feature Engineering, Statistics",
    },
    "JD-2 — Naver (Backend Engineer)": {
        "required": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
        "preferred": "REST API, CI/CD, Redis",
    },
    "JD-3 — Line (Frontend Engineer)": {
        "required": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
        "preferred": "Node.js, GraphQL, Redux, Jest, AWS",
    },
}

TOTAL_DOCS = len(RAW_RESUMES)  # 10


# ── Helpers ─────────────────────────────────────────────────────────────────
def silent_normalize(raw: str) -> list[str]:
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    result = normalize_skills(raw)
    sys.stdout = old_stdout
    return result


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    d = dot_product(a, b)
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return d / (na * nb)


def main():
    sep = "═" * 80
    thin = "─" * 80

    # ── 1. Normalize all resumes ────────────────────────────────────────
    resumes = [(name, silent_normalize(raw)) for name, raw in RAW_RESUMES]

    # ── 2. Build shared vocabulary ──────────────────────────────────────
    vocab_set = set()
    for _, skills in resumes:
        vocab_set.update(skills)
    vocabulary = sorted(vocab_set)
    V = len(vocabulary)
    skill_to_idx = {s: i for i, s in enumerate(vocabulary)}

    # ── 3. Compute df and IDF ───────────────────────────────────────────
    df = {}
    for skill in vocabulary:
        df[skill] = sum(1 for _, skills in resumes if skill in skills)

    idf = {skill: math.log(TOTAL_DOCS / df[skill]) for skill in vocabulary}

    # ── 4. Build resume TF-IDF vectors (length V) ──────────────────────
    resume_vectors = {}
    for name, skills in resumes:
        n = len(skills)
        tf = 1.0 / n
        vec = [0.0] * V
        for skill in skills:
            idx = skill_to_idx[skill]
            vec[idx] = tf * idf[skill]
        resume_vectors[name] = vec

    # ── 5. Build JD binary vectors (length V) ──────────────────────────
    jd_vectors = {}
    for jd_name, sections in JOB_DESCRIPTIONS.items():
        combined = sections["required"] + ", " + sections["preferred"]
        jd_skills = set(silent_normalize(combined))
        vec = [0.0] * V
        for skill in jd_skills:
            if skill in skill_to_idx:
                vec[skill_to_idx[skill]] = 1.0
        jd_vectors[jd_name] = vec

    # ── 6. Compute cosine similarity for all pairs ──────────────────────
    print(sep)
    print("  STAGE 5: COSINE SIMILARITY — RESUME × JD MATCHING")
    print(sep)

    all_rankings = {}

    for jd_name, jd_vec in jd_vectors.items():
        scores = []
        for name, res_vec in resume_vectors.items():
            sim = cosine_similarity(res_vec, jd_vec)
            scores.append((name, sim))

        # Sort: descending score, then alphabetically by first name on tie
        scores.sort(key=lambda x: (-round(x[1], 2), x[0].split()[0]))
        all_rankings[jd_name] = scores

    # ── 7. Print full ranked list per JD ────────────────────────────────
    for jd_name, scores in all_rankings.items():
        print(f"\n{thin}")
        print(f"  {jd_name}")
        print(thin)
        print(f"  {'Rank':>4s}  {'Candidate':<22s}  {'Score':>8s}  {'Bar'}")
        print(f"  {'─' * 4}  {'─' * 22}  {'─' * 8}  {'─' * 30}")

        for rank, (name, sim) in enumerate(scores, start=1):
            rounded = round(sim, 2)
            bar_len = int(rounded * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            marker = " ◀ TOP 3" if rank <= 3 else ""
            print(f"  {rank:>4d}  {name:<22s}  {rounded:>8.2f}  {bar}{marker}")

    # ── 8. Final summary: Top 3 per JD ──────────────────────────────────
    print(f"\n{sep}")
    print("  FINAL RESULTS — TOP 3 PER JD")
    print(sep)

    for jd_name, scores in all_rankings.items():
        top3 = scores[:3]
        formatted = ", ".join(f"{name}({round(sim, 2):.2f})" for name, sim in top3)
        print(f"\n  {jd_name}")
        print(f"  {formatted}")

    # ── 9. Cross-check: show which skills matched ───────────────────────
    print(f"\n{sep}")
    print("  MATCH DETAIL — TOP 1 PER JD (skill-level breakdown)")
    print(sep)

    for jd_name, scores in all_rankings.items():
        top_name = scores[0][0]
        top_score = scores[0][1]

        # Find matching skills
        jd_skills_set = set()
        sections = JOB_DESCRIPTIONS[jd_name]
        combined = sections["required"] + ", " + sections["preferred"]
        jd_skills_set = set(silent_normalize(combined)) & vocab_set

        resume_skills = set(
            s for _, skills in resumes if _ == top_name for s in skills
        )

        matched = sorted(jd_skills_set & resume_skills)
        missed = sorted(jd_skills_set - resume_skills)

        print(f"\n  {jd_name}")
        print(f"  Best match: {top_name} (cosine = {round(top_score, 2):.2f})")
        print(f"    ✔ Matched skills ({len(matched)}): {matched}")
        print(f"    ✘ Missing skills ({len(missed)}):  {missed}")

    print(f"\n{sep}")


if __name__ == "__main__":
    main()
