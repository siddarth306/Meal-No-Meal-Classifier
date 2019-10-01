import numpy as np

def parse(filename):
    r = np.genfromtxt(filename, delimiter=',', names=True, case_sensitive=True)
    return r
