import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	# Your code here
	if len(data) == 0 or degree < 0:
		return []

	x = np.asarray(data, dtype=float)

	return np.vander(x, N=degree + 1, increasing=True).tolist()