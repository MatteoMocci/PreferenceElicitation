import cvxpy as cp
import pandas as pd
import numpy as np
import json
import itertools
from utility import BOUNDARIES
from utility import most_uncertain, sum_of_margin, entropy
from time import perf_counter
from multiprocessing import Manager
from queue import Empty

# Number of criterias for the three capabilities
N_RESTORATIVENESS_CRITERIA = 5
N_NUTRITION_CRITERIA = 2
N_HEALTHCARE_CRITERIA = 5

# Enable / disable capabilities
RESTORATIVENESS = True
NUTRITION = False
HEALTHCARE = False

N_WORKERS = 8
PARALLELIZE = True

CATEGORIES = {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4}
MAX_ITER = 10
CRITERIAS = {
    "restorativeness": ["Physical activity and movement", "Enjoying aesthetic views", "Experiencing quietness", "Cultural activities", "Contact with nature"],
    "nutrition": ["Eating out", "Food provisioning for home consumption"],
    "care": ["Access to medical and health products", "Impatient care and rehabilitation", "Diagnosis and prevention", "Emergency services", "Social care and assistance"]
}

def build_model(learning_set, n_criteria, alpha=0.5):
    w = cp.Variable(n_criteria, nonneg=True)
    lamb = cp.Variable()
    mu = cp.Variable()
    xi = cp.Variable(len(learning_set), nonneg=True)

    constraints = [
        cp.sum(w) == 1,
        lamb >= 0.5,
        lamb <= 1,
    ]

    for i, (criteria, category) in enumerate(learning_set):
        criteria = np.asarray(criteria, dtype=float)

        if category > 0:
            votes = (criteria >= BOUNDARIES[category - 1]).astype(int)
            constraints.append(w @ votes + xi[i] >= lamb + mu)

        if category < len(BOUNDARIES):
            votes = (criteria >= BOUNDARIES[category]).astype(int)
            constraints.append(w @ votes - xi[i] <= lamb - mu)

    problem = cp.Problem(
        cp.Maximize(mu - alpha * cp.sum(xi)),
        constraints,
    )
    print("Solving model...", flush=True)
    start = perf_counter()
    problem.solve(solver=cp.HIGHS)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"Model fitting failed: {problem.status}")
    print(
        f"Model solved in {perf_counter() - start:.2f}s: "
        f"status={problem.status}, objective={problem.value}",
        f"Weights: {w.value}, lambda: {lamb.value}, mu: {mu.value}",
        flush=True,
    )

    return {
        "problem": problem,
        "constraints": constraints,
        "w": w,
        "lamb": lamb,
        "mu": mu,
        "xi": xi,
        "learning_set": learning_set,
        "n_criteria": n_criteria,
        "alpha": alpha
    }


def build_base_problems():
    df = pd.read_csv("learning_set.csv")
    df_r = df[df["capability"] == "restorativeness"]
    df_n = df[df["capability"] == "nutrition"]
    df_h = df[df["capability"] == "care"]

    learning_set_r = []
    learning_set_n = []
    learning_set_h = []

    for d in df_r, df_n, df_h:
        for _, row in d.iterrows():
            criteria = np.array(json.loads(row["criteria"]))
            category = row["category"]
            if d is df_r:
                learning_set_r.append((criteria, category))
            elif d is df_n:
                learning_set_n.append((criteria, category))
            else:
                learning_set_h.append((criteria, category))

    models = {}

    for name, enabled, learning_set, n_criteria in [
        ("restorativeness", RESTORATIVENESS, learning_set_r,
        N_RESTORATIVENESS_CRITERIA),
        ("nutrition", NUTRITION, learning_set_n, N_NUTRITION_CRITERIA),
        ("care", HEALTHCARE, learning_set_h, N_HEALTHCARE_CRITERIA),
    ]:
        
        if enabled:
            known_criteria = {
                tuple(criteria)
                for criteria, _ in learning_set
            }

            unclassified = [
                candidate
                for candidate in itertools.product(
                    *[BOUNDARIES + [0] for _ in range(n_criteria)]
                )
                if candidate not in known_criteria
            ]
            print(
                f"{name}: {len(learning_set)} labeled examples, "
                f"{len(unclassified)} candidates",
                flush=True,
            )
            
            model = build_model(learning_set, n_criteria)
            model["unclassified"] = unclassified
            models[name] = model
    return models


def preference_elicitation():
    problems = build_base_problems()
    strategy = "sum_of_margin" # possible values: "most_uncertain", "sum_of_margin", "entropy"
    print("Using strategy:", strategy, flush=True)
    for name, problem in problems.items():
        if problem is not None:
            k = 0
            unclassified = problem["unclassified"]

            
            while k < MAX_ITER and unclassified:
                print( f"\n{name}: iteration {k + 1}/{MAX_ITER}, {len(unclassified)} candidates remaining", flush=True)
                candidates = unclassified
                if PARALLELIZE:
                    bucket_size, remainder = divmod(len(unclassified), N_WORKERS)
                    buckets = []
                    start = 0
                    for i in range(N_WORKERS):
                        end = start + bucket_size + (1 if i < remainder else 0)
                        buckets.append(unclassified[start:end])
                        start = end
                    candidates = buckets
                parallel_options = {"parallelize": PARALLELIZE, "n_workers": N_WORKERS}
                args = (
                    candidates, problem["constraints"],
                    problem["w"], problem["lamb"], problem["mu"],
                )
                if strategy == "most_uncertain":
                    selected_index = most_uncertain(*args, **parallel_options)
                elif strategy == "sum_of_margin":
                    selected_index = sum_of_margin(
                        *args, problem["xi"], problem["alpha"], **parallel_options
                    )
                elif strategy == "entropy":
                    selected_index = entropy(
                        *args, problem["xi"], problem["alpha"], **parallel_options
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                if selected_index is None:
                    print("No more uncertain candidates to classify.", flush=True)
                    break
                unclassified_criteria = unclassified[selected_index]

                valid_answer = False
                while not valid_answer:
                    print(f"Please classify this alternative for {name}: ")
                    for criterion_name, value in zip(CRITERIAS[name], unclassified_criteria, strict=True):
                        print(f"  {criterion_name}: {value:.2f}")
                    user_input = input("Enter a category (0-4): ")
                    try:
                        category = int(user_input)
                        if category in range(5):
                            valid_answer = True
                        else:
                            print("Invalid input. Please enter a number between 0 and 4.")
                    except ValueError:
                        print("Invalid input. Please enter a number between 0 and 4.")
                
                criteria = np.asarray(unclassified_criteria, dtype=float)
                problem["learning_set"].append((criteria, category))

                updated_model = build_model(
                    problem["learning_set"],
                    problem["n_criteria"],
                    problem["alpha"],
                )

                problem.update(updated_model)
                unclassified.pop(selected_index)
                k += 1

if __name__ == "__main__":
    print("Starting preference elicitation...", flush=True)
    preference_elicitation()
