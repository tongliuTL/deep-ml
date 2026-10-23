import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    def get_percentiles(values):
        percentiles_dict = {}
        for p in percentiles:
            percentiles_dict[p] = round(float(np.percentile(values, p)), 2)
        return percentiles_dict

    e2e = np.sum(list(stage_latencies.values()), axis=0)
    e2e_mean = float(np.mean(e2e))
    means = {}
    stage_stats = {}

    for stage, values in stage_latencies.items():
        means[stage] = float(np.mean(values))
        stage_stats[stage] = {
            'mean': round(means[stage], 2),
            'std': round(float(np.std(values)), 2),
            'percentiles': get_percentiles(values),
        }

    stage_pct = {}

    for stage, mean in means.items():
        if e2e_mean:
            stage_pct[stage] = round(mean / e2e_mean * 100, 2)
        else:
            stage_pct[stage] = 0.0

    return {
        'e2e_mean': round(e2e_mean, 2),
        'e2e_percentiles': get_percentiles(e2e),
        'stage_stats': stage_stats,
        'bottleneck': max(means, key=means.get),
        'stage_pct': stage_pct, 
    }