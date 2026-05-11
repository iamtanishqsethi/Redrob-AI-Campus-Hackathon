"""
main.py — Full Pipeline Runner
--------------------------------
Runs all stages and prints the final submission answers.
"""

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
    print("Resume vs Job Description Matching Pipeline")
    print("===========================================\n")

    # Normalize all resumes
    print("Stage 1 & 2: Skill Normalization")
    print("--------------------------------")
    resumes = []
    for name, raw in RAW_RESUMES:
        skills = normalize_skills(raw)
        resumes.append((name, skills))
        print(f"  {name:<20s} -> {skills}")

    # TF-IDF
    print("\nStage 3: TF-IDF Computation")
    print("---------------------------")
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

    resume_vectors = {}
    for name, skills in resumes:
        n = len(skills)
        tf = 1.0 / n
        vec = [0.0] * V
        for skill in skills:
            vec[skill_to_idx[skill]] = tf * idf[skill]
        resume_vectors[name] = vec

    # JD Binary Vectors
    print("\nStage 4: JD Binary Vectors")
    print("--------------------------")
    jd_vectors = {}
    for jd_name, sections in JOB_DESCRIPTIONS.items():
        combined = sections["required"] + ", " + sections["preferred"]
        jd_skills = set(normalize_skills(combined))
        in_vocab = jd_skills & vocab_set
        outside = jd_skills - vocab_set
        vec = [0.0] * V
        for skill in in_vocab:
            vec[skill_to_idx[skill]] = 1.0
        jd_vectors[jd_name] = vec
        extra = f"  (outside vocab: {sorted(outside)})" if outside else ""
        print(f"  {jd_name}: {len(in_vocab)} active skills{extra}")

    # Cosine Similarity Ranking
    print("\nStage 5: Cosine Similarity Rankings")
    print("-----------------------------------")
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
        for rank, (name, sim) in enumerate(scores, start=1):
            print(f"  {rank:>2d}. {name:<22s} {round(sim,2):>5.2f}")

    print("\nSanity Checks")
    print("-------------")
    checks_passed = 0
    checks_total = 0

    def check(label, condition):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")

    from skill_normalizer import SKILL_ALIASES
    check('Vocabulary union matches', set(vocabulary) == vocab_set)
    check('All IDF values > 0', all(v > 0 for v in idf.values()))
    check('JD vectors are binary', all(all(x in (0.0,1.0) for x in v) for v in jd_vectors.values()))
    print(f"  Passed {checks_passed} out of {checks_total} checks.")

    print("\nFinal Results")
    print("=============")
    for jd_name, scores in all_rankings.items():
        top3 = scores[:3]
        formatted = ", ".join(f"{name} ({round(sim,2):.2f})" for name, sim in top3)
        print(f"\n  {jd_name}")
        print(f"  -> {formatted}")
    print()

if __name__ == "__main__":
    main()
