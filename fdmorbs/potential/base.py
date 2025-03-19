import numpy as np
class AbstractBasePotential:
    '''
    Class for time varying potentials
    '''
    def __init__(self, n_snapshots=1000, tmax=40., r_eval=0.001):
        self.n_snapshots = n_snapshots
        self.tmax = tmax
        self.r_eval = r_eval
        self.timesteps = np.linspace(0, tmax, n_snapshots)




    