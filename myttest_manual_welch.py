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

s = np.sqrt( ( var_a/len(a) + var_b/len(b) ) )

t = (mean_a - mean_b) / s

v1 = len(a) - 1
v2 = len(b) - 1
N1 = len(a)
N2 = len(b)

df_numerator = ( (var_a/len(a)) + (var_b/len(b)) ) ** 2

df_denominator = ( var_a ** 2 / ( (N1**2) *v1) ) + ( var_b ** 2 / ( (N2**2) *v2) )

df = df_numerator / df_denominator


p = (1 - stats.t.cdf(np.abs(t), df=df)) * 2

print("Manual Welch Test t:\t", t, "p:\t", p)

t2,p2 = stats.ttest_ind(a,b)

print("t2:\t", t2, "p2:\t", p2)
