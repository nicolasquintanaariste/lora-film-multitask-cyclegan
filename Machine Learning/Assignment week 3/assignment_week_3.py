import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as train_test_split

df = pd.read_csv("week3.csv", header=None, comment="#")

print(df.head())

# Extract columns
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

###################################################################################################################################
# Question a)
###################################################################################################################################
# i) Plot data

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_title('Week 3 Data Scatter Plot')

plt.show() 


# ii) Polynomial Features and Lasso Regression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=5)

X_poly = poly.fit_transform(X)
print(X_poly)
feature_names = poly.get_feature_names_out(['x1', 'x2'])
print(feature_names)

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

lasso = {}
C = [1, 100, 10000, 1000000]
for c in C:
    model = Lasso(alpha=1/c)
    lasso[c] = model.fit(X_poly, y)
    print(f"Lasso model with alpha={c}:")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_}")

print(feature_names)


# iii) Plot data with model predictions
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_model_surface(ax, model, poly):
    grid = np.linspace(-1.5, 1.5, 100)
    X1, X2 = np.meshgrid(grid, grid)
    Y_pred = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            poly_features = poly.transform([[X1[i, j], X2[i, j]]])
            Y_pred[i, j] = model.predict(poly_features)[0]

    ax.plot_surface(X1, X2, Y_pred, cmap='viridis', alpha=0.7)

for c, model in lasso.items():
    y_pred = model.predict(X_poly)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, color='r', alpha=0.5, s=20)
    plot_model_surface(ax, model, poly)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.set_title(f'Lasso Model Prediction, c={c}')


    legend_elements = [
        Patch(facecolor='#ADFF2F', label='Prediction Surface'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Training Data')
    ]
    ax.legend(handles=legend_elements)

plt.show()


###################################################################################################################################
# Question i) e) REPEAT FOR RIDGE REGRESSION
###################################################################################################################################
# e) Rigde Regression
from sklearn.linear_model import Ridge

ridge = {}
C = [1, 10, 100, 1000]
for c in C:
    model = Ridge(alpha=1/c)
    ridge[c] = model.fit(X_poly, y)
    print(f"Ridge model with alpha={c}:")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_}")                      


# iii) Plot data with model predictions

def plot_model_surface(ax, model, poly):
    grid = np.linspace(-1.5, 1.5, 100)
    X1, X2 = np.meshgrid(grid, grid)
    Y_pred = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            poly_features = poly.transform([[X1[i, j], X2[i, j]]])
            Y_pred[i, j] = model.predict(poly_features)[0]

    ax.plot_surface(X1, X2, Y_pred, cmap='viridis', alpha=0.7)

for c, model in ridge.items():
    y_pred = model.predict(X_poly)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, color='r', alpha=0.5, s=20)
    plot_model_surface(ax, model, poly)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.set_title(f'Ridge Model Prediction, c={c}')


    legend_elements = [
        Patch(facecolor='#ADFF2F', label='Prediction Surface'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Training Data')
    ]
    ax.legend(handles=legend_elements)

plt.show()


###################################################################################################################################
# Question ii) 
###################################################################################################################################
# a) Error on Lasso regression with cross-validation
from sklearn.model_selection import cross_val_score
import statistics as stats

cv_scores = []
C = np.arange(1, 60, 2)
for c in C:
    model = Lasso(alpha=1/c)
    cv_scores.append(cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error'))

mean = [stats.mean(-score) for score in cv_scores]
std = [stats.stdev(score) for score in cv_scores]

plt.errorbar(C, mean, yerr=std, fmt='o', capsize=5)
plt.xlabel('C')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Regression Cross-Validation Error')
plt.grid()
plt.show()


###################################################################################################################################
# Question ii) c) 
###################################################################################################################################
# a) Error on Ridge regression with cross-validation
cv_scores = []
C = np.arange(0.1, 5, 0.1)
for c in C:
    model = Ridge(alpha=1/c)
    cv_scores.append(cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error'))

mean = [stats.mean(-score) for score in cv_scores]
std = [stats.stdev(score) for score in cv_scores]

plt.errorbar(C, mean, yerr=std, fmt='o', capsize=5)
plt.xlabel('C')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression Cross-Validation Error')
plt.grid()
plt.show()