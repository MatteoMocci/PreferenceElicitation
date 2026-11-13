import cvxpy as cp
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from queue import Empty

BOUNDARIES = [0.2, 0.4, 0.6, 0.8]
# The boundaries for the three capabilities (flat - same for each criteria)


def process_bucket(bucket, eval_f, progress_queue, *args):
    # Fresh CVXPY IDs prevent collisions with variables created in this worker.
    # Use one mapping so constraints and evaluator arguments share the same copies.
    replacements = {}
    for arg in args:
        expressions = arg if isinstance(arg, list) else [arg]
        for expression in expressions:
            if isinstance(expression, (cp.Expression, cp.constraints.constraint.Constraint)):
                for variable in expression.variables():
                    if id(variable) not in replacements:
                        replacements[id(variable)] = cp.Variable(
                            variable.shape, **variable.attributes
                        )
    args = tuple(
        [constraint.tree_copy(replacements) for constraint in arg]
        if isinstance(arg, list) else
        arg.tree_copy(replacements) if isinstance(arg, cp.Expression) else arg
        for arg in args
    )
    templates = create_templates(*args)
    results = []
    for x in bucket:
        results.append(eval_f(x, *args, templates=templates))
        progress_queue.put(1)
    return results

def count_possible_categories(x, constraints, w, lamb, mu, *, templates=None):
    if templates is None:
        templates = create_templates(constraints, w, lamb, mu)
    return sum(
        is_compatible(x, category, constraints, w, lamb, mu, templates=templates)
        for category in range(len(BOUNDARIES) + 1)
    )

def mr_sort(x,w,lamb):
    for i, b in enumerate(reversed(BOUNDARIES)):
        votes = (x >= b).astype(int)
        score = w @ votes
        if score >= lamb:
            return len(BOUNDARIES) - i
    return 0

def create_template(category_type, constraints, w, lamb, mu, *,
                    simulation=False, xi=None, alpha=0.5):
    votes = {}
    extra_constraints = []
    slack = cp.Variable(nonneg=True) if simulation else 0

    if category_type != "lowest":
        votes["lower"] = cp.Parameter(w.size)
        extra_constraints.append(w @ votes["lower"] + slack >= lamb + mu)

    if category_type != "highest":
        votes["upper"] = cp.Parameter(w.size)
        extra_constraints.append(w @ votes["upper"] - slack <= lamb - mu)

    objective = mu - alpha * (cp.sum(xi) + slack) if simulation else mu
    problem = cp.Problem(cp.Maximize(objective), constraints + extra_constraints)
    assert problem.is_dcp(dpp=True)
    return problem, votes


def create_templates(constraints, w, lamb, mu, xi=None, alpha=0.5):
    category_types = ("lowest", "middle", "highest")
    templates = {"compatibility": {
        kind: create_template(kind, constraints, w, lamb, mu)
        for kind in category_types
    }}
    if xi is not None:
        templates["simulation"] = {
            kind: create_template(kind, constraints, w, lamb, mu,
                                  simulation=True, xi=xi, alpha=alpha)
            for kind in category_types
        }
    return templates


def solve_template(x, category, templates):
    kind = ("lowest" if category == 0 else
            "highest" if category == len(BOUNDARIES) else "middle")
    problem, votes = templates[kind]
    x = np.asarray(x, dtype=float)
    if "lower" in votes:
        votes["lower"].value = (x >= BOUNDARIES[category - 1]).astype(float)
    if "upper" in votes:
        votes["upper"].value = (x >= BOUNDARIES[category]).astype(float)
    problem.solve(solver=cp.HIGHS, enforce_dpp=True, warm_start=False)
    return problem


def is_compatible(x, test_category, base_constraints, w, lamb, mu, *, templates=None):
    if templates is None:
        templates = create_templates(base_constraints, w, lamb, mu)
    test_problem = solve_template(x, test_category, templates["compatibility"])
    if test_problem.status == cp.OPTIMAL and mu.value is not None and mu.value >= 1e-6:
        return True
    else:
        return False

def most_uncertain(unclassified, constraints, w, lamb, mu, *, parallelize=False, n_workers=8):
    uncertainty = {}
    with tqdm(
        total=(sum(map(len, unclassified)) if parallelize else len(unclassified)) * 5,
        desc="Checking candidate categories",
        unit="solve",
    ) as progress:
        if not parallelize:
            templates = create_templates(constraints, w, lamb, mu)
            for i, x in enumerate(unclassified):
                uncertainty[i] = 0
                for test_category in range(5):
                    if is_compatible(x, test_category, constraints, w, lamb, mu,
                                     templates=templates):
                        uncertainty[i] += 1
                    progress.update(1)
        else:
            with Manager() as manager:
                progress_queue = manager.Queue()
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [
                        executor.submit(
                            process_bucket,
                            bucket,
                            count_possible_categories,
                            progress_queue,
                            constraints, w, lamb, mu,
                        )
                        for bucket in unclassified
                    ]

                counts = []
                for future in futures:
                    bucket_counts = future.result()
                    counts.extend(bucket_counts)
                    progress.update(len(bucket_counts) * 5)
            uncertainty = dict(enumerate(counts))
    n_ambiguous = sum(count >= 2 for count in uncertainty.values())
    print(
        f"Alternatives with two or more possible categories: {n_ambiguous}",
        flush=True,
    )
    if n_ambiguous == 0:
        return None
    return max(uncertainty, key=uncertainty.get)

def simulate_assignment(x, test_category, base_constraints, w, lamb, mu, xi, alpha,
                        *, templates=None):
    if templates is None:
        templates = create_templates(base_constraints, w, lamb, mu, xi, alpha)
    test_problem = solve_template(x, test_category, templates["simulation"])
    if test_problem.status != cp.OPTIMAL:
        raise RuntimeError(
            f"Hypothetical assignment failed: {test_problem.status}"
        )
    return float(test_problem.value)

def evaluate_assignment(x, constraints, w, lamb, mu, xi, alpha, *, templates=None):
    if templates is None:
        templates = create_templates(constraints, w, lamb, mu, xi, alpha)
    objectives = []
    n_possible = 0
    for category in range(len(BOUNDARIES) + 1):
        objectives.append(simulate_assignment(
            x, category, constraints, w, lamb, mu, xi, alpha, templates=templates
        ))
        if is_compatible(x, category, constraints, w, lamb, mu, templates=templates):
            n_possible += 1
    return n_possible, objectives


def evaluate_alternatives(unclassified, constraints, w, lamb, mu, xi, alpha,
                          parallelize, n_workers, progress):
    """Evaluate possible categories and objectives, preserving alternative order."""
    args = (constraints, w, lamb, mu, xi, alpha)
    solves_per_alternative = (len(BOUNDARIES) + 1) * 2
    if parallelize:
        with Manager() as manager:
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(
                        process_bucket, bucket, evaluate_assignment, progress_queue, *args
                    )
                    for bucket in unclassified
                ]
                while not all(future.done() for future in futures):
                    try:
                        completed = progress_queue.get(timeout=0.1)
                        progress.update(completed * solves_per_alternative)
                    except Empty:
                        pass
                # Workers may finish before the last progress messages are read.
                while True:
                    try:
                        completed = progress_queue.get_nowait()
                        progress.update(completed * solves_per_alternative)
                    except Empty:
                        break
                # Submission order preserves original indices and first-match ties.
                for future in futures:
                    yield from future.result()
    else:
        templates = create_templates(*args)
        for x in unclassified:
            result = evaluate_assignment(x, *args, templates=templates)
            progress.update(solves_per_alternative)
            yield result


def sum_of_margin(unclassified, constraints, w, lamb, mu, xi, alpha, *,
                  parallelize=False, n_workers=8):
    scores = {}
    n_categories = len(BOUNDARIES) + 1

    with tqdm(
        total=(sum(map(len, unclassified)) if parallelize else len(unclassified)) * n_categories * 2,
        desc="Computing sum of margins",
        unit="solve",
    ) as progress:
        for index, (n_possible, objectives) in enumerate(evaluate_alternatives(
            unclassified, constraints, w, lamb, mu, xi, alpha,
            parallelize, n_workers, progress,
        )):
            if n_possible >= 2:
                scores[index] = sum(objectives)

    print(
        f"Alternatives with two or more possible categories: {len(scores)}",
        flush=True,
    )

    if not scores:
        return None

    selected_index = max(scores, key=scores.get)
    print(
        f"Selected sum of margins: {scores[selected_index]:.6f}",
        flush=True,
    )
    return selected_index

def entropy(unclassified, constraints, w, lamb, mu, xi, alpha, *,
            parallelize=False, n_workers=8):
    scores = {}
    n_categories = len(BOUNDARIES) + 1

    with tqdm(
        total=(sum(map(len, unclassified)) if parallelize else len(unclassified)) * n_categories * 2,
        desc="Computing entropy",
        unit="solve",
    ) as progress:
        for index, (n_possible, objectives) in enumerate(evaluate_alternatives(
            unclassified, constraints, w, lamb, mu, xi, alpha,
            parallelize, n_workers, progress,
        )):
            if n_possible < 2:
                continue

            positive_objectives = np.maximum(
                np.asarray(objectives, dtype=float), 0.0
            )
            total = positive_objectives.sum()
            if total == 0.0:
                scores[index] = 0.0
                continue

            probabilities = positive_objectives / total
            positive_probabilities = probabilities[probabilities > 0.0]
            scores[index] = float(
                -np.sum(
                    positive_probabilities * np.log(positive_probabilities)
                )
            )

    print(
        f"Alternatives with at least two categories accepted by "
        f"is_compatible: {len(scores)}",
        flush=True,
    )
    
    if not scores:
        return None

    selected_index = max(scores, key=scores.get)
    print(
        f"Selected entropy: {scores[selected_index]:.6f}",
        flush=True,
    )
    return selected_index
