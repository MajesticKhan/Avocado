# import libraries
import numpy as np

np.set_printoptions(linewidth=300)


def MRP(P: np.ndarray, R: np.ndarray, Beta: float) -> np.ndarray:
    """
    The Bellman Equation used to find value states in a Markov Reward Process given:
    - States
    - Probability Matrix
    - Reward
    - beta

    For more details see David Silverman's lecture:
    - https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf

    We can find the values for each state using the equation below
    V = R + Beta*P*V
    Solve for V, V = (I - beta*P)^-1 * R

    :param P: Probability Matrix (NxN)
    :param R: Immediate or expected Reward for each states (Nx1)
    :param Beta: Scalar value representing the discount factor
    :return: A Nx1 column vector representing the expected reward values for each state
    """

    # Create Identity matrix
    I = np.identity(R.shape[0])

    # Solve for V, with beta being .98
    V = np.matmul(np.linalg.inv(I - (Beta * P)), R)

    return V
