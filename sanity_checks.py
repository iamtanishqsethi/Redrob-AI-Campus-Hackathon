"""
Stage 6 — Sanity Checks
-------------------------
Validates the entire pipeline: normalization, vocabulary, IDF, vectors, JDs.
Standard library only.
"""

import io
import math
import sys
from skill_normalizer import normalize_skills, SKILL_ALIASES

# ── Data (identical to previous stages) ─────────────────────────────────────
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


def silent_normalize(raw: str) -> list[str]:
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    result = normalize_skills(raw)
    sys.stdout = old_stdout
    return result


def main():
    sep = "═" * 80
    thin = "─" * 80
    passed = 0
    failed = 0
    total = 0

    def check(label: str, condition: bool, detail: str = ""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✔ PASS  {label}")
        else:
            failed += 1
            print(f"  ✘ FAIL  {label}")
        if detail:
            print(f"          {detail}")

    # ── Rebuild all pipeline state ──────────────────────────────────────
    resumes = [(name, silent_normalize(raw)) for name, raw in RAW_RESUMES]

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
            vec[skill_to_idx[skill]] = tf * idf[skill]
        resume_vectors[name] = vec

    jd_vectors = {}
    jd_skills_map = {}
    for jd_name, sections in JOB_DESCRIPTIONS.items():
        combined = sections["required"] + ", " + sections["preferred"]
        jd_skills = set(silent_normalize(combined))
        jd_skills_map[jd_name] = jd_skills
        vec = [0.0] * V
        for skill in jd_skills:
            if skill in skill_to_idx:
                vec[skill_to_idx[skill]] = 1.0
        jd_vectors[jd_name] = vec

    # ════════════════════════════════════════════════════════════════════
    # CHECK 1: Normalization aliases
    # ════════════════════════════════════════════════════════════════════
    print(sep)
    print("  CHECK 1: NORMALIZATION ALIASES")
    print(sep)

    # 1a. "Sklearn" → machine_learning
    check(
        '"Sklearn" → machine_learning',
        SKILL_ALIASES.get("sklearn") == "machine_learning",
        f'SKILL_ALIASES["sklearn"] = {SKILL_ALIASES.get("sklearn")!r}',
    )

    # 1b. "kubernates" → kubernetes
    check(
        '"kubernates" → kubernetes',
        SKILL_ALIASES.get("kubernates") == "kubernetes",
        f'SKILL_ALIASES["kubernates"] = {SKILL_ALIASES.get("kubernates")!r}',
    )

    # 1c. "data-viz" → data_visualization
    check(
        '"data-viz" → data_visualization',
        SKILL_ALIASES.get("data_viz") == "data_visualization",
        f'SKILL_ALIASES["data_viz"] = {SKILL_ALIASES.get("data_viz")!r}',
    )

    # 1d. "matplotlib" → data_visualization
    check(
        '"matplotlib" → data_visualization',
        SKILL_ALIASES.get("matplotlib") == "data_visualization",
        f'SKILL_ALIASES["matplotlib"] = {SKILL_ALIASES.get("matplotlib")!r}',
    )

    # 1e. R04 (Sneha Patel) dedup: "data-viz" and "matplotlib" both → data_visualization
    sneha_skills = dict(resumes)["Sneha Patel"]
    dv_count = sneha_skills.count("data_visualization")
    check(
        "R04 dedup: data-viz + matplotlib → single data_visualization",
        dv_count == 1,
        f'Sneha Patel skills = {sneha_skills}  (data_visualization appears {dv_count}x)',
    )

    # 1f. Verify end-to-end on specific inputs
    test_cases = [
        ("Sklearn",    ["machine_learning"]),
        ("kubernates", ["kubernetes"]),
        ("data-viz",   ["data_visualization"]),
        ("matplotlib", ["data_visualization"]),
    ]
    for raw_in, expected in test_cases:
        result = silent_normalize(raw_in)
        check(
            f'normalize("{raw_in}") → {expected}',
            result == expected,
            f"Got: {result}",
        )

    # ════════════════════════════════════════════════════════════════════
    # CHECK 2: Vocabulary integrity
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  CHECK 2: VOCABULARY INTEGRITY")
    print(sep)

    # 2a. Size
    print(f"  Vocabulary size: {V}")
    check(
        f"Vocabulary size = {V} (positive, non-zero)",
        V > 0,
    )

    # 2b. Vocabulary matches union of all resume skills
    union = set()
    for _, skills in resumes:
        union.update(skills)
    check(
        "Vocabulary == union of all resume skills",
        set(vocabulary) == union,
        f"|vocabulary| = {len(vocabulary)}, |union| = {len(union)}, "
        f"diff = {set(vocabulary).symmetric_difference(union) or '∅'}",
    )

    # 2c. All vocabulary entries are alphabetically sorted
    check(
        "Vocabulary is sorted alphabetically",
        vocabulary == sorted(vocabulary),
    )

    # 2d. No duplicates in vocabulary
    check(
        "No duplicates in vocabulary",
        len(vocabulary) == len(set(vocabulary)),
    )

    # ════════════════════════════════════════════════════════════════════
    # CHECK 3: IDF values
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  CHECK 3: IDF VALUES")
    print(sep)

    # Highest IDF = skills with df=1 → IDF = ln(10)
    max_idf_expected = math.log(TOTAL_DOCS)
    max_idf_skills = [s for s in vocabulary if df[s] == 1]
    actual_max_idf = max(idf.values())

    check(
        f"Highest IDF = ln(10) = {max_idf_expected:.6f}",
        abs(actual_max_idf - max_idf_expected) < 1e-9,
        f"Actual max IDF = {actual_max_idf:.6f}",
    )
    print(f"          Skills with highest IDF (df=1): {len(max_idf_skills)} skills")
    for s in max_idf_skills:
        print(f"            • {s}")

    # Lowest IDF = skill with highest df
    min_idf_skill = min(idf, key=idf.get)
    min_idf_value = idf[min_idf_skill]
    min_idf_df = df[min_idf_skill]
    expected_min_idf = math.log(TOTAL_DOCS / min_idf_df)

    check(
        f'Lowest IDF: "{min_idf_skill}" (df={min_idf_df}, IDF={min_idf_value:.6f})',
        abs(min_idf_value - expected_min_idf) < 1e-9,
        f"Expected ln({TOTAL_DOCS}/{min_idf_df}) = {expected_min_idf:.6f}",
    )

    # All IDFs are positive (no skill appears in all 10 resumes → df < 10 → IDF > 0)
    all_positive = all(v > 0 for v in idf.values())
    check(
        "All IDF values are positive (no skill in all 10 resumes)",
        all_positive,
    )

    # ════════════════════════════════════════════════════════════════════
    # CHECK 4: Resume vector bounds
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  CHECK 4: RESUME VECTOR BOUNDS")
    print(sep)

    any_over_1 = False
    for name, vec in resume_vectors.items():
        max_comp = max(vec)
        over = max_comp > 1.0
        if over:
            any_over_1 = True
        check(
            f"{name:<20s}  max component = {max_comp:.6f}  (≤ 1.0)",
            not over,
        )

    check(
        "No resume vector has any component > 1.0",
        not any_over_1,
    )

    # Also verify vector lengths are all V
    all_correct_len = all(len(v) == V for v in resume_vectors.values())
    check(
        f"All resume vectors have length {V}",
        all_correct_len,
    )

    # ════════════════════════════════════════════════════════════════════
    # CHECK 5: JD binary vectors — no 1s outside vocabulary
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  CHECK 5: JD BINARY VECTOR INTEGRITY")
    print(sep)

    for jd_name, vec in jd_vectors.items():
        # 5a. Vector length
        check(
            f"{jd_name}  vector length = {len(vec)} (should be {V})",
            len(vec) == V,
        )

        # 5b. Only 0s and 1s
        only_binary = all(v in (0.0, 1.0) for v in vec)
        check(
            f"{jd_name}  all components are 0 or 1",
            only_binary,
        )

        # 5c. Every 1 corresponds to a vocab skill
        active_indices = [i for i, v in enumerate(vec) if v == 1.0]
        active_skills = [vocabulary[i] for i in active_indices]
        all_in_vocab = all(s in vocab_set for s in active_skills)
        check(
            f"{jd_name}  all 1-bits are in vocabulary ({len(active_skills)} active)",
            all_in_vocab,
            f"Active: {active_skills}",
        )

        # 5d. No JD skill that normalized successfully is missing from vector
        jd_canonical = jd_skills_map[jd_name]
        in_vocab_skills = jd_canonical & vocab_set
        missing_from_vec = [s for s in in_vocab_skills if vec[skill_to_idx[s]] != 1.0]
        check(
            f"{jd_name}  no in-vocab JD skill missing from vector",
            len(missing_from_vec) == 0,
            f"Missing: {missing_from_vec}" if missing_from_vec else "",
        )

        # 5e. Report skills that normalized but aren't in vocab (expected for pytorch, redis)
        outside_vocab = jd_canonical - vocab_set
        if outside_vocab:
            print(f"          ℹ Skills normalized but outside resume vocabulary "
                  f"(correctly excluded): {sorted(outside_vocab)}")

    # ════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  SANITY CHECK SUMMARY")
    print(sep)
    print(f"  Total checks: {total}")
    print(f"  Passed:       {passed}  ✔")
    print(f"  Failed:       {failed}  {'✘' if failed else ''}")
    print()
    if failed == 0:
        print("  🎉 ALL CHECKS PASSED — Pipeline is consistent end-to-end.")
    else:
        print(f"  ⚠  {failed} CHECK(S) FAILED — Review above for details.")
    print(sep)


if __name__ == "__main__":
    main()
