import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("/home/tugberkozdemir/Workspace/tarf/data/temp.csv")

bill1 = df["PAY_AMT1"]

bill1.hist(bins=1000)
plt.xlim(0,20000)
plt.show()