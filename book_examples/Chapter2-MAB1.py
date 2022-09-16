# to realize MAB problem
# import libs
import numpy as np


# define class 
class BernoulliBandit:
    # init function
    def __init__(self, K):
        # K numbers to stand the possibility for each trail(K times) 
        self.probs = np.random.uniform(size = K)

        # get the id number of the max possibility
        self.best_id  = np.argmax(self.probs)
        # get the max possibility
        self.max_prob = self.probs[self.best_id]

        # get param K to the class
        self.K = K

    # function for each trail's final
    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1  # trail is a success
        else:
            return 0  # trail is a failure


np.random.seed(1)  # 设定随机种子,使实验具有可重复性
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
      (bandit_10_arm.best_id, bandit_10_arm.max_prob))

