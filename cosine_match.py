"""
Stage 5 — Cosine Similarity: Resume × JD Matching
---------------------------------------------------
Builds full-length TF-IDF vectors for resumes and binary vectors for JDs
over the shared 50-skill vocabulary, then ranks candidates per JD using
cosine similarity.  Standard library only.
"""

import math
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
    print("Stage 5: Cosine Similarity - Resume x JD Matching")
    print("=================================================\n")

    resumes = [(name, normalize_skills(raw)) for name, raw in RAW_RESUMES]

    vocab_set = set()
    for _, skills in resumes:
        vocab_set.update(skills)
    vocabulary = sorted(vocab_set)
    V = len(vocabulary)
    skill_to_idx = {s: i for i, s in enumerate(vocabulary)}

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
            idx = skill_to_idx[skill]
            vec[idx] = tf * idf[skill]
        resume_vectors[name] = vec

    jd_vectors = {}
    for jd_name, sections in JOB_DESCRIPTIONS.items():
        combined = sections["required"] + ", " + sections["preferred"]
        jd_skills = set(normalize_skills(combined))
        vec = [0.0] * V
        for skill in jd_skills:
            if skill in skill_to_idx:
                vec[skill_to_idx[skill]] = 1.0
        jd_vectors[jd_name] = vec

    all_rankings = {}

    for jd_name, jd_vec in jd_vectors.items():
        scores = []
        for name, res_vec in resume_vectors.items():
            sim = cosine_similarity(res_vec, jd_vec)
            scores.append((name, sim))

        scores.sort(key=lambda x: (-round(x[1], 2), x[0].split()[0]))
        all_rankings[jd_name] = scores

    for jd_name, scores in all_rankings.items():
        print(f"\n{jd_name}")
        print("-" * len(jd_name))
        for rank, (name, sim) in enumerate(scores, start=1):
            print(f"  {rank:>2d}. {name:<22s} Score: {round(sim, 2):.2f}")

    print("\nFinal Results - Top 3 Per JD")
    print("============================\n")

    for jd_name, scores in all_rankings.items():
        top3 = scores[:3]
        formatted = ", ".join(f"{name} ({round(sim, 2):.2f})" for name, sim in top3)
        print(f"{jd_name}")
        print(f"  -> {formatted}\n")

    print("\nMatch Detail - Top 1 Per JD (Skill-level breakdown)")
    print("===================================================\n")

    for jd_name, scores in all_rankings.items():
        top_name = scores[0][0]
        top_score = scores[0][1]

        sections = JOB_DESCRIPTIONS[jd_name]
        combined = sections["required"] + ", " + sections["preferred"]
        jd_skills_set = set(normalize_skills(combined)) & vocab_set

        resume_skills = set(
            s for _, skills in resumes if _ == top_name for s in skills
        )

        matched = sorted(jd_skills_set & resume_skills)
        missed = sorted(jd_skills_set - resume_skills)

        print(f"{jd_name}")
        print(f"Best match: {top_name} (cosine = {round(top_score, 2):.2f})")
        print(f"  Matched skills: {matched}")
        print(f"  Missing skills: {missed}\n")


if __name__ == "__main__":
    main()
