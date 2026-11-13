"""Run serial/parallel parity checks with real HiGHS solves on Windows."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from unittest.mock import patch

import numpy as np

import main
import utility


def check():
    model = main.build_model(
        [(np.full(2, value), category)
         for category, value in enumerate([0, .2, .4, .6, .8])],
        2,
    )
    candidates = [(0.2, 0.8), (0, 0), (0.4, 0.6), (0.8, 0.2), (1, 1)]
    # Uneven buckets plus an empty bucket exercise flattening and indexing.
    buckets = [candidates[:2], candidates[2:], []]
    args = tuple(model[key] for key in ('constraints', 'w', 'lamb', 'mu', 'xi', 'alpha'))
    serial = [utility.evaluate_assignment(x, *args) for x in candidates]
    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(utility.process_bucket, b, utility.evaluate_assignment,
                                       progress_queue, *args)
                       for b in buckets]
            parallel = [result for future in futures for result in future.result()]
    for (count_s, values_s), (count_p, values_p) in zip(serial, parallel, strict=True):
        assert count_s == count_p
        np.testing.assert_allclose(values_s, values_p, rtol=1e-9, atol=1e-9)
    for strategy in (utility.most_uncertain, utility.sum_of_margin, utility.entropy):
        strategy_args = args[:4] if strategy is utility.most_uncertain else args
        sequential = strategy(candidates, *strategy_args)
        parallel = strategy(buckets, *strategy_args, parallelize=True, n_workers=2)
        assert sequential == parallel, (strategy.__name__, sequential, parallel)
        assert strategy([[], []], *strategy_args, parallelize=True, n_workers=2) is None
        print(f'{strategy.__name__}: serial and parallel selections match')

    # Exercise selection/removal and rebuilding buckets across two iterations.
    with patch.object(main, 'N_WORKERS', 2), patch.object(main, 'MAX_ITER', 2), \
         patch.object(main, 'RESTORATIVENESS', False), \
         patch.object(main, 'NUTRITION', True), patch.object(main, 'HEALTHCARE', False), \
         patch('builtins.input', return_value='2'):
        main.preference_elicitation()
    print('All parity and two-iteration integration checks passed.')


if __name__ == '__main__':
    check()
