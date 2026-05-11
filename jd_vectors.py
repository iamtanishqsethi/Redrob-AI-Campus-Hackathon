"""
Stage 4 — JD Binary Vectors
-----------------------------
Normalizes 3 Job Descriptions through the SAME alias pipeline used for
resumes, then builds binary vectors against the shared vocabulary from
Stage 3.  Standard library only.
"""

from skill_normalizer import normalize_skills

# ── Resume data (needed to rebuild vocabulary) ──────────────────────────────
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

# ── JD data ─────────────────────────────────────────────────────────────────
JOB_DESCRIPTIONS = {
    "JD1 — Kakao ML Engineer": {
        "required": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
        "preferred": "NLP, BERT, Feature Engineering, Statistics",
    },
    "JD2 — Naver Backend Engineer": {
        "required": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
        "preferred": "REST API, CI/CD, Redis",
    },
    "JD3 — Line Frontend Engineer": {
        "required": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
        "preferred": "Node.js, GraphQL, Redux, Jest, AWS",
    },
}





def main():
    vocab_set = set()
    for _, raw in RAW_RESUMES:
        vocab_set.update(normalize_skills(raw))
    vocabulary = sorted(vocab_set)

    print("Stage 4: JD Binary Vectors")
    print("==========================")
    print(f"Shared vocabulary size: {len(vocabulary)} skills\n")

    for jd_name, sections in JOB_DESCRIPTIONS.items():
        print(f"\n{jd_name}")
        print("-" * len(jd_name))

        combined_raw = sections["required"] + ", " + sections["preferred"]
        print(f"Raw (combined): {combined_raw!r}\n")

        all_skills = normalize_skills(combined_raw)

        req_skills = set(normalize_skills(sections["required"]))
        pref_skills = set(normalize_skills(sections["preferred"]))
        all_skills_set = set(all_skills)

        # Check for any skills that were discarded (not in alias map)
        raw_tokens = [t.strip().lower() for t in combined_raw.split(",")]
        discarded = [t for t in raw_tokens
                     if t and t not in [s for s in all_skills]]
        # More precise: check which raw tokens didn't produce a canonical match
        # We'll note skills not in vocabulary instead
        not_in_vocab = [s for s in all_skills if s not in vocab_set]
        in_vocab = [s for s in all_skills if s in vocab_set]

        print(f"\nCanonical skills: {all_skills}")
        print(f"  - Required:  {sorted(req_skills)}")
        print(f"  - Preferred: {sorted(pref_skills)}")
        if not_in_vocab:
            print(f"  - NOT in resume vocabulary (ignored): {not_in_vocab}")
        print(f"  - In vocabulary: {len(in_vocab)}/{len(all_skills)}")

        # Build binary vector
        vector = []
        for skill in vocabulary:
            vector.append(1 if skill in all_skills_set else 0)
        jd_vectors[jd_name] = vector

        print(f"\nBinary vector ({sum(vector)} active / {len(vector)} total):")
        print(f"{'Idx':>5s}  {'Skill':<35s}  {'Bit':>3s}  {'Source':>10s}")
        print(f"{'-' * 5}  {'-' * 35}  {'-' * 3}  {'-' * 10}")
        for i, skill in enumerate(vocabulary):
            bit = vector[i]
            if bit == 1:
                src = "Required" if skill in req_skills else "Preferred"
                marker = "1"
            else:
                src = ""
                marker = "0"
            print(f"{i:>5d}  {skill:<35s}  {marker:>3s}  {src:>10s}")

        print()

    print("\nSide-by-Side: All JD Binary Vectors")
    print("===================================")

    jd_names = list(jd_vectors.keys())
    short_names = ["JD1:ML", "JD2:BE", "JD3:FE"]

    header = f"{'Idx':>3s}  {'Skill':<30s}"
    for sn in short_names:
        header += f"  {sn:>7s}"
    print(header)
    print(f"{'-' * 3}  {'-' * 30}" + "  ------- " * 3)

        row = f"{i:>3d}  {skill:<30s}"
        active_any = False
        for jd_name in jd_names:
            bit = jd_vectors[jd_name][i]
            if bit:
                active_any = True
            row += f"  {'  1':>7s}" if bit else f"  {'  0':>7s}"
        if active_any:
            row += "  <"
        print(row)

    print("\nJD Overlap Analysis")
    print("===================")

    for i in range(len(jd_names)):
        for j in range(i + 1, len(jd_names)):
            v1 = jd_vectors[jd_names[i]]
            v2 = jd_vectors[jd_names[j]]
            active_1 = {vocabulary[k] for k, b in enumerate(v1) if b}
            active_2 = {vocabulary[k] for k, b in enumerate(v2) if b}
            overlap = active_1 & active_2
            print(f"  {short_names[i]} ∩ {short_names[j]}: "
                  f"{len(overlap)} shared → {sorted(overlap) if overlap else '∅'}")

    print()


if __name__ == "__main__":
    main()
