import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm


x1 = [0, 1, 2, 2, 3]
x2 = [0, 1, 3, 0, 4]
r = ['A', 'A', 'B', 'A', 'B']

dict = {"x1":x1, "x2":x2, "r":r}

df = pd.DataFrame(dict)

plt.scatter(df['x1'], df['x2'])
plt.show()


data = df[["x1", "x2"]].values

result = df['r']

model = svm.SVC(kernel="linear")

model.fit(data, result)

print(model.predict([[3, 3]]))