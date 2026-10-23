import math

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    # Your code here
    values = sorted(latencies)

    def percentile(p):
        if not values:
            return 0.0

        n = len(values)
        position = (n - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower

        result = values[lower] + weight * (values[upper] - values[lower])

        return round(result, 4)

    return {
        "P50": percentile(0.50),
        "P95": percentile(0.95),
        "P99": percentile(0.99),
    }