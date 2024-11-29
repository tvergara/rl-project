import numpy as np
from tqdm import tqdm

def run_experiment(method, n=100):
    sample_result = method()
    num_lists = len(sample_result)
    aggregated_results = [np.zeros_like(lst, dtype=float) for lst in sample_result]

    for i in range(num_lists):
        aggregated_results[i] += np.array(sample_result[i], dtype=float)

    for _ in tqdm(range(1, n)):
        results = method()
        for i in range(num_lists):
            aggregated_results[i] += np.array(results[i], dtype=float)

    averaged_results = [lst / n for lst in aggregated_results]
    return averaged_results
