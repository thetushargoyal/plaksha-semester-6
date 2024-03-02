import numpy as np

def linevo(A, p, n):
    """
    Calculates the final probabilistic state after multiple applications of the operator.

    Args:
    - A: Probabilistic operator representing the experiment (numpy array)
    - p: Initial probabilistic state (numpy array)
    - n: Number of times the experiment is conducted (integer)

    Returns:
    - q: Final probabilistic state (numpy array)
    """
    q = np.linalg.matrix_power(A, n).dot(p)
    # Ensure q is a valid probabilistic state by normalizing it
    q /= np.sum(q)
    return q

# Define the probabilistic operator A and the initial state p
A = np.array([[0.00, 0.00],
              [0.00, 0.00],])

p = np.array([1, 0])

# Number of times the experiment is conducted
n = 3

# Calculate the final probabilistic state
q = linevo(A, p, n)

print("Final probabilistic state after", n, "applications of the operator:")
print(q)
