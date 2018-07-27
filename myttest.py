import os
import numpy as np
from scipy import stats
import pandas as pd

os.chdir("/home/ekele/machine_learning_examples/ab_testing/")

data = pd.read_csv("advertisement_clicks.csv")

a = data[data['advertisement_id']=='A']
b = data[data['advertisement_id']=='B']

a = a['action']
b = b['action']

mean_a, mean_b = a.mean(), b.mean()

var_a, var_b = a.var(ddof=1), b.var(ddof=1)

N = len(a)

df = 2*N - 2

s = np.sqrt( (var_a + var_b) / 2 )

t = (a.mean() - b.mean()) / ( s * np.sqrt(2 / N) )

p = (1 - stats.t.cdf(np.abs(t), df=df)) * 2

print("t:\t", t, "p:\t", p)

t2,p2 = stats.ttest_ind(a,b)

print("t2:\t", t2, "p2:\t", p2)
