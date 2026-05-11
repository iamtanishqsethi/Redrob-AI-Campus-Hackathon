"""
main.py — Full Pipeline Runner
--------------------------------
Runs all stages and prints the final submission answers.
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

TOTAL_DOCS = len(RAW_RESUMES)


def silent_normalize(raw):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    result = normalize_skills(raw)
    sys.stdout = old_stdout
    return result


def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a, b):
    d = dot_product(a, b)
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return d / (na * nb)


def main():
    sep = "═" * 70
    thin = "─" * 70

    print(sep)
    print("  REDROB AI CAMPUS HACKATHON — RESUME × JD MATCHING PIPELINE")
    print(sep)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 1 & 2: Normalize all resumes
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'▸ STAGE 1 & 2: SKILL NORMALIZATION':─<70}")
    resumes = []
    for name, raw in RAW_RESUMES:
        skills = silent_normalize(raw)
        resumes.append((name, skills))
        print(f"  {name:<20s} → {skills}")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 3: TF-IDF
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'▸ STAGE 3: TF-IDF COMPUTATION':─<70}")

    vocab_set = set()
    for _, skills in resumes:
        vocab_set.update(skills)
    vocabulary = sorted(vocab_set)
    V = len(vocabulary)
    skill_to_idx = {s: i for i, s in enumerate(vocabulary)}

    print(f"  Shared vocabulary: {V} unique canonical skills")

    df = {}
    for skill in vocabulary:
        df[skill] = sum(1 for _, skills in resumes if skill in skills)
    idf = {skill: math.log(TOTAL_DOCS / df[skill]) for skill in vocabulary}

    min_idf_skill = min(idf, key=idf.get)
    max_idf_val = max(idf.values())
    print(f"  Lowest  IDF: \"{min_idf_skill}\" (df={df[min_idf_skill]}, IDF={idf[min_idf_skill]:.4f})")
    print(f"  Highest IDF: ln(10) = {max_idf_val:.4f}  ({sum(1 for s in vocabulary if df[s]==1)} skills with df=1)")

    resume_vectors = {}
    for name, skills in resumes:
        n = len(skills)
        tf = 1.0 / n
        vec = [0.0] * V
        for skill in skills:
            vec[skill_to_idx[skill]] = tf * idf[skill]
        resume_vectors[name] = vec

    # ════════════════════════════════════════════════════════════════════
    # STAGE 4: JD Binary Vectors
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'▸ STAGE 4: JD BINARY VECTORS':─<70}")

    jd_vectors = {}
    for jd_name, sections in JOB_DESCRIPTIONS.items():
        combined = sections["required"] + ", " + sections["preferred"]
        jd_skills = set(silent_normalize(combined))
        in_vocab = jd_skills & vocab_set
        outside = jd_skills - vocab_set
        vec = [0.0] * V
        for skill in in_vocab:
            vec[skill_to_idx[skill]] = 1.0
        jd_vectors[jd_name] = vec
        extra = f"  (outside vocab: {sorted(outside)})" if outside else ""
        print(f"  {jd_name}: {len(in_vocab)} active skills{extra}")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 5: Cosine Similarity Ranking
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'▸ STAGE 5: COSINE SIMILARITY RANKINGS':─<70}")

    all_rankings = {}
    for jd_name, jd_vec in jd_vectors.items():
        scores = []
        for name, res_vec in resume_vectors.items():
            sim = cosine_similarity(res_vec, jd_vec)
            scores.append((name, sim))
        scores.sort(key=lambda x: (-round(x[1], 2), x[0].split()[0]))
        all_rankings[jd_name] = scores

    for jd_name, scores in all_rankings.items():
        print(f"\n  {jd_name}")
        print(f"  {'Rank':>4s}  {'Candidate':<22s}  {'Score':>6s}")
        print(f"  {'─'*4}  {'─'*22}  {'─'*6}")
        for rank, (name, sim) in enumerate(scores, start=1):
            marker = " ◀" if rank <= 3 else ""
            print(f"  {rank:>4d}  {name:<22s}  {round(sim,2):>6.2f}{marker}")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 6: Sanity Checks
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'▸ STAGE 6: SANITY CHECKS':─<70}")

    checks_passed = 0
    checks_total = 0

    def check(label, condition):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
        status = "✔" if condition else "✘"
        print(f"  {status} {label}")

    from skill_normalizer import SKILL_ALIASES
    check('"Sklearn" → machine_learning',       SKILL_ALIASES.get("sklearn") == "machine_learning")
    check('"kubernates" → kubernetes',           SKILL_ALIASES.get("kubernates") == "kubernetes")
    check('"data-viz" → data_visualization',     SKILL_ALIASES.get("data_viz") == "data_visualization")
    check('"matplotlib" → data_visualization',   SKILL_ALIASES.get("matplotlib") == "data_visualization")
    check(f'Vocabulary = {V} = union of all resume skills', set(vocabulary) == vocab_set)
    check('All IDF values > 0',                  all(v > 0 for v in idf.values()))
    check('No resume vector component > 1.0',    all(max(v) <= 1.0 for v in resume_vectors.values()))
    check('All JD vectors are binary {0,1}',     all(all(x in (0.0,1.0) for x in v) for v in jd_vectors.values()))

    sneha_skills = dict(resumes)["Sneha Patel"]
    check('R04 dedup: data-viz + matplotlib → 1 data_visualization',
          sneha_skills.count("data_visualization") == 1)

    print(f"\n  Result: {checks_passed}/{checks_total} checks passed")

    # ════════════════════════════════════════════════════════════════════
    # FINAL SUBMISSION ANSWERS
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ★  FINAL RESULTS ★")
    print(sep)

    for jd_name, scores in all_rankings.items():
        top3 = scores[:3]
        formatted = ", ".join(f"{name}({round(sim,2):.2f})" for name, sim in top3)
        print(f"\n  {jd_name}")
        print(f"  {formatted}")

    print(f"\n{sep}")


if __name__ == "__main__":
    main()
