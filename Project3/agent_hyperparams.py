class Params:
    def __init__(self):
        self.eps_start = 0.9
        self.eps_end = 0.025
        self.batch_sz = 32

        self.replay_mem = 10_000
        self.st_replay_mem = 5_000
        self.update_steps = 5_000

        self.learn_rate = 1e-4
        self.gamma = 0.99