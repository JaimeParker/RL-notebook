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

S = ["s1", "s2", "s3", "s4", "s5"]  # state group
# action group
A = ["hold s1", "go s1", "go s2", "go s3", "go s4", "go s5", "probably to go"]

# state transition matrix (function)
# different dut to actions
P = {
    "s1-hold s1-s1":  1.0,
    "s1-goto s2-s2":  1.0,
    "s2-goto s1-s1":  1.0,
    "s2-goto s3-s3":  1.0,
    "s3-goto s4-s4":  1.0,
    "s3-goto s5-s5":  1.0,
    "s4-goto s5-s5":  1.0,
    "s4-probably goto-s2": 0.2,
    "s4-probably goto-s3": 0.4,
    "s4-probably goto-s4": 0.4
}

# reward function
R = {
    "s1-hold s1": -1,
    "s1-goto s2": 0,
    "s2-goto s1": -1,
    "s2-goto s3": -2,
    "s3-goto s4": -2,
    "s3-goto s5": 0,
    "s4-goto s5": 10,
    "s4-probably goto": 1,
}

MDP_sample = (S, A, P, R, gamma)

# strategy 1
Pi_1 = {
    "s1-hold s1": 0.5,
    "s1-goto s2": 0.5,
    "s2-goto s1": 0.5,
    "s2-goto s3": 0.5,
    "s3-goto s4": 0.5,
    "s3-goto s5": 0.5,
    "s4-goto s5": 0.5,
    "s4-probably goto": 0.5,
}

# strategy 2
Pi_2 = {
    "s1-hold s1": 0.6,
    "s1-goto s2": 0.4,
    "s2-goto s1": 0.3,
    "s2-goto s3": 0.7,
    "s3-goto s4": 0.5,
    "s3-goto s5": 0.5,
    "s4-goto s5": 0.1,
    "s4-probably goto": 0.9,
}


def join(str1, str2):
    return str1 + '-' + str2


def sample(mdp, strategy, timestep_max, number):
    state, action, state_mat, reward, discount = mdp


