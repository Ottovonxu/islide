import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse 
import pdb


parser = argparse.ArgumentParser()

parser.add_argument("--file", action="store", dest="file", type=str, required=True)
parser.add_argument("--print_configs", action="store_true", dest="print_configs", default=False)
parser.add_argument("--key", action="store", dest="key", type=str, default="auc")
parser.add_argument("--configs", action="store", dest="configs", type=str, default="")
parser.add_argument("--use_log_scale", action="store_true", default=False)
parser.add_argument("--iter_per_epoch", action="store", type=int, required=True)
results = parser.parse_args()
configs = results.configs.split(",")
agg_file = results.file
key = results.key
use_log_scale = results.use_log_scale
iter_per_epoch = results.iter_per_epoch

agg_data = pd.read_csv(agg_file)
agg_data = agg_data.loc[agg_data.loc[:,"value"] != -1]
if (results.print_configs):
  configs = agg_data.loc[:,"config"].unique()
  for c in configs:
    print(c)
  exit(0)


maxL = -1
maxx = []

def create_legend(s):
  ss = s.split("_")
  lb = ss[0] + '_' + ss[1] + '_'+ ss[2] + '_'+ ss[-1]
  if ss[0] == "MH":
    lb = ss[0] + "_" + ss[2] + "_" + ss[4] + '_' + ss[-1]
  return lb

for cfg in configs:
  df = agg_data.loc[(agg_data.loc[:, "config"] == cfg) & (agg_data.loc[:, "key"] == key), :]
  df.loc[:,"x_axis"] = df.loc[:,"epoch"] * iter_per_epoch + df.loc[:,"iteration"]
  df.sort_values("x_axis", inplace=True)
  x_axis = df.loc[:,"x_axis"]
  if use_log_scale:
    plt.plot(range(0, len(df.loc[:,"value"])), df.loc[:,"value"], label=create_legend(cfg))
    #xlabels = [str(x) for x in x_axis]
  else:
    plt.plot(x_axis, df.loc[:,"value"], label=create_legend(cfg))
  if maxL < len(df.loc[:,"value"]):
    maxL = len(df.loc[:,"value"])
    maxx = df["x_axis"]
if use_log_scale:
  plt.xticks(range(0, maxL), maxx, rotation='vertical')
plt.legend(loc="best")
plt.show()