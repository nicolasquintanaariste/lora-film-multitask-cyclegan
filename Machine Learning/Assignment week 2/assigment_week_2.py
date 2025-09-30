import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("week2.csv", header=None, comment="#")

print(df.head())

# Extract columns
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

colours = {-1: "red", 1: "blue"}
markers = {-1: 'o', 1: '+'}
# Plot data
for key, val in colours.items():
    marker_style = '+' if key == 1 else '.'
    plt.scatter(X1[y == key], X2[y == key], c=val, marker = markers[key], label=str(key))

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Week 2 data")
plt.legend(title="Class, y")
plt.show()