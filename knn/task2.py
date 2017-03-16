#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    v = np.loadtxt(filename)
    print(v)
    loo = v[:,1]
    kk = v[:,0]
    plt.plot(kk, loo)
    plt.show()