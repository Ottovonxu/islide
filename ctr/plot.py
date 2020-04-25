import matplotlib.pyplot as plt
import numpy as np
import sys
from os.path import dirname, abspath, join
import argparse

cur_dir = dirname(abspath(__file__))
pltname = sys.argv[1]
filenames = sys.argv[2:]
topk=None
skipK=None
movP=None
IterationCt =100

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')



fig = plt.figure()
ax = plt.subplot(111)

for fn in filenames:
    print(fn)
    fnn = fn.split('/')[-1]
    with open(fn, "r") as f:
        lines = f.readlines()
    vs = [float(l.strip()) for l in lines]
    bs = [i.strip('BS') for i in fn.split('_') if 'BS' in i]
    if bs == []:
      bs = 1024
    else:
      bs = int(bs[0])
    IterationCt = bs
    print(IterationCt, "<- iteration ct")
    itr = np.arange(0, len(vs)) * IterationCt
    if topk is not None:
        vs = vs[0:topk]
        itr = itr[0:topk]
    if skipK is not None:
        vs = vs[skipK:]
        itr = itr[skipK:]
    if movP is not None:
      vs = moving_average(vs, periods=movP)
      itr = moving_average(itr, periods=movP)
    
    ax.plot(itr, vs, label=fnn)


box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.4,
                 box.width, box.height * 0.6])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=1)
#plt.legend(loc='best')
plt.savefig(pltname)
