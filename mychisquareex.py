import os
import numpy as np
from scipy.stats import chi2, chi2_contingency
import pandas as pd

os.chdir("/home/ekele/machine_learning_examples/ab_testing/")

data = pd.read_csv("advertisement_clicks.csv")

a = data[data['advertisement_id']=='A']
b = data[data['advertisement_id']=='B']

a = a['action']
b = b['action']

def get_p_value(T):
    det = T[0,0]*T[1,1] - T[0,1]*T[1,0]
    c2 = float(det) / T[0].sum() * det / T[1].sum() * T.sum() / T[:,0].sum() / T[:,1].sum()
    p = 1 - chi2.cdf(x=c2,df=1)
    return p

A_click = a.sum()
A_noclick = a.size - a.sum()

B_click = b.sum()
B_noclick = b.size - b.sum()

T = np.array([ [A_click, A_noclick], [B_click, B_noclick] ])
print(get_p_value(T))
