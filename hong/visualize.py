import numpy as np
import pandas as pd

df = pd.read_csv('/home/hong/PRProject/data/voice.csv')
df.head()

print("Total number of samples:{}".format(df.shape[0]))
print("Number of male: {}".format(df[df.label == 'male'].shape[0]))
print("Number of female: {}".format(df[df.label == 'female'].shape[0]))


import seaborn
seaborn.pairplot(df[['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR','label']], hue='label', size=2)
seaborn.plt.show()


