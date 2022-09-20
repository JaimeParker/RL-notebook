# to realize MAB problem
# use Epsilon Greedy algorithm

# import libs
import numpy as np
import matplotlib.pylab as plt


# class to compose a Bernouli Bandit model, create bandit
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
            # cautions, this should be less equal than probs[k]
            return 1  # trail is a success
        else:
            return 0  # trail is a failure


# class to solver the Bernouli Bandit problem
# created bandit solver
class SolverBandit:
    # init function
    def __init__(self, bandit):
        # get bandit instance from BernouliBandit
        self.bandit = bandit
        # set conut for each bandit's trial
        self.counts = np.zeros(self.bandit.K)
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

    # define for override
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


# define class for greedy algrothim, inherit from class SolverBandit
class EpsilonGreedy(SolverBandit):
    # init function
    def __init__(self, bandit, epsilon = 0.01, init_prob = 1.0):
        # inherit from it's parent class
        # there a good example telling about super()
        # https://www.runoob.com/python/python-func-super.html
        super().__init__(bandit)
        self.epsilon = epsilon
        # setting all options's prob as value of init_prob
        self.estimates = np.array([init_prob] * self.bandit.K)

    # epsilon greedy algorithm's main process 
    # be advised here, function with the same name of function in 
    # class SolverBandit
    def run_one_step(self):
        # the algorithm formula
        if np.random.random() < self.epsilon:
            # choose one number randomly form 0 to K
            k = np.random.randint(0, self.bandit.K)
        else:
            # choose the max prob from estimated probs
            k = np.argmax(self.estimates)

        # update the estimates array
        r = self.bandit.step(k)  # acquire the final
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (
            r - self.estimates[k])  # use 1.0 to keep float

        # why return k?
        return k


# define plot function
def plot_results(solvers, solver_names):
    # the use of enumerate can be referred here:
    # https://www.geeksforgeeks.org/enumerate-in-python/
    # this is set for different epsilons' situtions
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

np.random.seed(1)  # make random numbers be the same

# set 10 arms for Bernouli Bandit
K = 10
bandit_10_arm = BernoulliBandit(K)
print("create %d arms Bernouli Bandit" % K)
print("the max probility is No. %d, with %.4f" %
      (bandit_10_arm.best_id, bandit_10_arm.max_prob))

np.random.seed(1)  # make random numbers be the same
# create greedy solver instance
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-greedy cumulative regret：', 
    epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])