# to realize MAB problem
# import libs
import numpy as np
import matplotlib.pylab as plt


# class to compose a Bernouli Bandit model
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
        if np.random.rand() > self.probs[k]:
            return 1  # trail is a success
        else:
            return 0  # trail is a failure


np.random.seed(1)  # make random numbers be the same

# set 10 arms for Bernouli Bandit
K = 10
bandit_10_arm = BernoulliBandit(K)
print("create %d arms Bernouli Bandit" % K)
print("the max probility is No. %d, with %.4f" %
      (bandit_10_arm.best_id, bandit_10_arm.max_prob))


# class to solver the Bernouli Bandit problem
class SolverBandit:
    # init function
    def __init__(self, bandit):
        # get bandit instance from BernouliBandit
        self.bandit = bandit
        # set conut for each bandit's trial
        self.counts = np.zerons(self.bandit.K)
        # set regret for current step
        self.regret = 0
        # set a list to load regrets
        self.regrets = []
        # set a list to load actions
        self.actions = []

    # calculate cumulatice regret
    def update_regret(self, k):
        # update regret by the formula R(a) = Q* - Q(a)
        self.regret = self.regret + self.bandit.max_prob - self.bandit.probs[k]
        self.regrets.append(self.regret) # add it to the regrets list

    # 
    def run_one_step(self):
        # In user defined base classes, abstract methods should raise this 
        # exception when they require derived classes to override the method, 
        # or while the class is being developed to indicate that the real 
        # implementation still needs to be added.
        raise NotImplementedError

    
    # main iteration process
    def run(self, num_steps):
        # num_steps means the iteration times
        for _ in range (num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)



