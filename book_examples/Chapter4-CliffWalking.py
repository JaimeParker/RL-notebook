import copy


class CliffWalkingEnv:
    """悬崖漫步环境"""
    def __init__(self, col=12, row=4):
        self.col = col
        self.row = row
        # transition matrix P[state][action]
        self.P = self.create_p()

    def create_p(self):
        p_mat = [[[] for _ in range(4)] for _ in range(self.row * self.col)]

        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        for i in range(self.row):
            for j in range(self.col):
                for a in range(4):

                    if i == self.row - 1 and j > 0:
                        p_mat[i * self.col + j][a] = [(1, i * self.col + j, 0, True)]
                        continue

                    next_x = min(self.col - 1, max(0, j + change[a][0]))
                    next_y = min(self.row - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.col + next_x

                    reward = -1
                    done = False

                    if next_y == self.row - 1 and next_x > 0:
                        done = True
                        if next_x != self.col - 1:
                            reward = -100

                    p_mat[i * self.col + j][a] = [(1, next_state, reward, done)]

        return p_mat
