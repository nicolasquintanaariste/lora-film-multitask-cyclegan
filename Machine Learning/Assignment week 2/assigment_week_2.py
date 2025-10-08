import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as train_test_split

df = pd.read_csv("week2.csv", header=None, comment="#")

print(df.head())

# Extract columns
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

print(X, y)

###################################################################################################################################
# Question a)
###################################################################################################################################
# i) Plot data
train_colours = {-1: "green", 1: "orange"}
train_markers = {-1: '.', 1: '+'}
train_labels = {-1: "Class -1", 1: "Class +1"}
def plot_scatter(X1,X2, y, colours, markers, labels, title):
    for key, val in colours.items():
        marker_style = '+' if key == 1 else '.'
        plt.scatter(X1[y == key], X2[y == key], c=val, marker = markers[key], label=str(labels[key]))

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.legend(title="Class, y")

plot_scatter(X[:,0], X[:,1], y, train_colours, train_markers, train_labels, "Week 2 Data")
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split.train_test_split(X, y, test_size=0.1, random_state=13)

# ii) Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print("\nLogistic Regression")
print("Coefficients (w1, w2):", np.round(log_reg.coef_[0], 3))
print("Intercept (w0):", np.round(log_reg.intercept_[0], 3))

# iii) Prediction and evaluation

from sklearn.metrics import accuracy_score, f1_score

y_pred_reg = log_reg.predict(X_test)

test_colours = {-1: "red", 1: "blue"}
test_labels = {-1: "Pred -1", 1: "Pred +1"}
plot_scatter(X_test[:,0], X_test[:,1], y_pred_reg, test_colours, train_markers, test_labels, "Logistic Regression")

print("Test Accuracy:",  accuracy_score(y_test, y_pred_reg))
print("Test f1 score:", f1_score(y_test, y_pred_reg))

# Plot decision boundary
def plot_decision_boundary(model):
    w1, w2 = model.coef_[0]
    b = model.intercept_[0]

    slope = -w1 / w2
    intercept = -b / w2

    plt.axline((0, intercept), slope=slope, color='black', linestyle='--', label="Decision boundary")

plot_decision_boundary(log_reg)
plt.legend()
plt.show()

###################################################################################################################################
# Question b)
###################################################################################################################################
# i) Linear SVM
from sklearn.svm import LinearSVC

# Train SVM with different C values

svms = {}

print("\nLinear SVMs")

C = [0.001, 0.01, 1, 100]
for c in C:
    svm = LinearSVC(random_state=13, C=c)
    svm.fit(X_train, y_train)
    svms[c] = svm

    print("\nSVM with C =", c)
    print("Coefficients (w1, w2):", np.round(svms[c].coef_[0], 3))
    print("Intercept (w0):", np.round(svms[c].intercept_[0], 3))

    # Prediction
    y_pred_svm = svms[c].predict(X_test)
    print("Accuracy: ", round(accuracy_score(y_test, y_pred_svm),2), " f1 score: ", round(f1_score(y_test, y_pred_svm),2))

    # Plot
    # plot_scatter(X_train[:,0], X_train[:,1], y_train, train_colours, train_markers, train_labels, "Training Data")
    plot_scatter(X_test[:,0], X_test[:,1], y_pred_svm, test_colours, train_markers, test_labels, f"Linear SVM, C={c}")
    plot_decision_boundary(svms[c])
    plt.show()

###################################################################################################################################
# Question c)
###################################################################################################################################
# i) Add square features
X_train_squared = np.column_stack((X_train, X_train[:, 0] ** 2, X_train[:, 1] ** 2))

log_reg_sqr = LogisticRegression()
log_reg_sqr.fit(X_train_squared, y_train)

print("\nLogistic Regression with 4 features")
print("Coefficients (w1, w2, w3, w4):", np.round(log_reg_sqr.coef_[0], 3))
print("Intercept (w0):", np.round(log_reg_sqr.intercept_[0], 3))


# ii) Prediction
X_test_squared = np.column_stack((X_test, X_test[:, 0] ** 2, X_test[:, 1] ** 2))
y_pred_reg_sqr = log_reg_sqr.predict(X_test_squared)

test_colours = {-1: "red", 1: "blue"}
test_labels = {-1: "Pred -1", 1: "Pred +1"}
plot_scatter(X_train[:,0], X_train[:,1], y_train, train_colours, train_markers, train_labels, "Training Data")
plot_scatter(X_test[:,0], X_test[:,1], y_pred_reg_sqr, test_colours, train_markers, test_labels, "Quadratic Logistic Regression")
#plt.show()

print("Test Accuracy:", round(accuracy_score(y_test, y_pred_reg_sqr), 2))
print("Test f1 score:", round(f1_score(y_test, y_pred_reg_sqr), 2))

# iii) Baseline predictor
most_common_class = 1 if np.sum(y_train == 1) > np.sum(y_train == -1) else -1

y_pred_baseline = np.full_like(y_test, most_common_class)

print("\nBaseline Predictor")
print("Test Accuracy:", round(accuracy_score(y_test, y_pred_baseline), 2))
print("Test f1 score:", round(f1_score(y_test, y_pred_baseline), 2))

# iv) Plot decision boundary.
def plot_decision_boundary_quadratic(model):
    w1, w2, w3, w4 = model.coef_[0]
    w0 = model.intercept_[0]
    
    x1 = np.linspace(-0.75, 0.75, 300)
    x2 = np.linspace(-1, 1, 300)

    X1, X2 = np.meshgrid(x1, x2)

    Z = w0 + w1*X1 + w2*X2 + w3*X1**2 + w4*X2**2

    plt.contour(X1, X2, Z, levels=[0], colors='black', linestyles='--', labels="Decision boundary")

plot_decision_boundary_quadratic(log_reg_sqr)
plt.show()


