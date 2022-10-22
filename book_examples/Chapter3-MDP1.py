# describe here
# Markov process

# import libs
import numpy as np

# set random seed for a certain outcome
np.random.seed(0)

# state transition matrix
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

P = np.array(P)  # convert to numpy datatype

rewards = [-1, -2, -2, 10, 1, 0]  # reward function
gamma = 0.5


# return or gain
def compute_return(start_index, chain, discount):
    gain = 0  # reward

    for i in reversed(range(start_index, len(chain))):
        gain = discount * gain + rewards[chain[i] - 1]

    return gain


# testing a state episode
test_chain = [1, 2, 3, 6]
test_start_index = 0
G = compute_return(test_start_index, test_chain, gamma)
print("return by this test episode is %s" % G)


# get reward by matrix
def compute(state_mat, reward_func, discount):
    states_num = state_mat.shape[0]  # shape[0] = shape[1]
    reward = np.array(reward_func).reshape((-1, 1))  # convert to column vector
    value = (np.linalg.inv(np.eye(states_num) - discount * P))
    value = np.matmul(value, reward)

    return value


V = compute(P, rewards, gamma)
print("each state value of MRP is: \n", V)
