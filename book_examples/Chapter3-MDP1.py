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


def sample(MDP, Pi, timestep_max, number):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态,开始接下来的循环
        episodes.append(episode)
    return episodes


# 采样5次,每个序列最长不超过20步
episodes = sample(MDP_sample, Pi_1, 20, 5)
print('第一条序列\n', episodes[0])
print('第二条序列\n', episodes[1])
print('第五条序列\n', episodes[4])

# 怪了 pycharm里报局部变量的错，jupyter里没问题

