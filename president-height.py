#https://muratdemirci.me/2023/04/02/python-ile-290-makine-ogrenimi-projesi/
#  Importing numpy and pandas
import numpy as np
import pandas as pd

data = pd.read_csv("../data/president_heights.csv")
# print(data.head())

height = np.array(data["height(cm)"])
# print(height)

# print("Mean of heights =", height.mean())
# print("Standard Deviation of height =", height.std())
# print("Minimum height =", height.min())
# print("Maximum height =", height.max())

# print("25th percentile =", np.percentile(height, 25))
# print("Median =", np.median(height))
# print("75th percentile =", np.percentile(height, 75))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.hist(height)
plt.title("Height Distribution of Presidents of USA")
plt.xlabel("height(cm)")
plt.ylabel("Number")
plt.show()


