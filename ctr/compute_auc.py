import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

import sys

if len(sys.argv) < 2:
  print ("usage: python3 <cmd> labels.txt predictions.txt")
  exit(0)
file1 = sys.argv[1]
file2 = sys.argv[2]

y = pd.read_csv(file1, index_col=False, header=None)[0].values
pred = pd.read_csv(file2, index_col=False, header=None)[0].values


fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
auc = metrics.auc(fpr, tpr)
print(auc)
plt.plot(fpr, tpr, label=file2.split("/")[-1])
plt.legend(loc="best")
plt.savefig(file2 + ".png")
