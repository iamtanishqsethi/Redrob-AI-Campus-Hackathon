"""
Stage 2 — Process 10 Resumes
------------------------------
Uses the normalize_skills function from skill_normalizer.py
to process each candidate's raw skill string and print
intermediate steps + final canonical skill lists.
"""

from skill_normalizer import normalize_skills

# ── Resume data ─────────────────────────────────────────────────────────────
RESUMES = [
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


def main():
    separator = "═" * 72

    print(separator)
    print("  SKILL NORMALIZATION — 10 RESUMES")
    print(separator)

    all_results = {}

    for idx, (name, raw_skills) in enumerate(RESUMES, start=1):
        print(f"\n{'─' * 72}")
        print(f"  [{idx:02d}] {name}")
        print(f"  Raw: {raw_skills!r}")
        print(f"{'─' * 72}")

        normalized = normalize_skills(raw_skills)
        all_results[name] = normalized

        print(f"\n  ✅ {name}: {normalized}\n")

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + separator)
    print("  SUMMARY")
    print(separator)
    for name, skills in all_results.items():
        print(f"  {name:<20s} → {skills}")

    # ── Validation checks ───────────────────────────────────────────────
    print("\n" + separator)
    print("  VALIDATION CHECKS")
    print(separator)

    checks = [
        ("Karan Mehta",   "Sklearn → machine_learning",    "machine_learning",    all_results["Karan Mehta"]),
        ("Rahul Gupta",   "kubernates → kubernetes",        "kubernetes",          all_results["Rahul Gupta"]),
        ("Aditya Kumar",  "TypeScrpit → typescript",        "typescript",          all_results["Aditya Kumar"]),
        ("Sneha Patel",   "matplotlib → data_visualization","data_visualization",  all_results["Sneha Patel"]),
    ]

    all_passed = True
    for candidate, label, expected, skills in checks:
        passed = expected in skills
        status = "✔ PASS" if passed else "✘ FAIL"
        print(f"  {status}  {candidate}: {label}  (in {skills})")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 All validation checks passed!")
    else:
        print("  ❌ Some checks failed — review alias map.")

    print(separator)


if __name__ == "__main__":
    main()
