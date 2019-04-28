from scipy.optimize import minimize
import numpy as np
import numpy.linalg as la
import numpy.random as random
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import time

random.seed(37)

# Verify example from 17.2
d = 2
k = 4
true = random.randn(d, k)
true = normalize(true, axis=0)

m = 420
train_pts = random.randn(d, m)
train_pts = train_pts / max(la.norm(train_pts, axis=0))
# train_pts = normalize(train_pts, axis=0)
train_labels = true.T @ train_pts
train_labels = np.argmax(train_labels, axis=0)

m_test = int(0.2 * m)
test_pts = random.randn(d, m_test)
test_pts = test_pts / max(la.norm(test_pts, axis=0))
# test_pts = normalize(test_pts, axis=0)
test_labels = true.T @ test_pts
test_labels = np.argmax(test_labels, axis=0)

def delta(i, j):
    """
    Zero-one loss
    """
    if i != j:
        return 1
    return 0

def svm_loss(lmbda, data, labels):
    """
    Let d be the ambient dimension
    w :         np vector of size d * k
    lmbda :     regularization parameter
    data :      np array of size d by m
    labels :    np vector of size m
    """
    def loss(w):
        dk = w.shape[0]
        d, m = data.shape

        loss = lmbda * (w @ w)

        # Reshape w to matrix
        w = w.reshape(d, int(dk / d))
        
        for j in range(m):
            largest = -np.Infinity
            for i in range(k):
                curr = delta(i, labels[j]) + w[:, i] @ data[:, j]
                if curr > largest:
                    largest = curr
            largest = largest - w[:, labels[j]] @ data[:, j]
            loss += (1 / m) * largest
        return loss
    return loss

def multiclass_svm(lmbda, pts, labels):
    loss_fn = svm_loss(lmbda, pts, labels)
    w_guess = random.randn(d * k)
    res = minimize(loss_fn, w_guess)
    loss_val = loss_fn(res.x)
    return res.x, loss_val

def tune(pts, labels, start=-5, end=1, num=7):
    """
    Use the provided pts and associated labels to determine the
    best lambda to run multiclass svm
    """
    lmbdas = np.logspace(start, end, num)
    best_loss = np.Infinity
    best_lmbda = -1
    for lmbda in lmbdas:
        _, loss = multiclass_svm(lmbda, pts, labels)
        if loss < best_loss:
            best_loss = loss
            best_lmbda = lmbda
    return lmbda

def predict(w, pts):
    """
    Predict the labels associated with the points.
    w :     np vector of size d * k
    pts :   np array of size d * n where n is number of points
    """
    dk = w.shape[0]
    d, n = pts.shape
    w = w.reshape(d, int(dk / d))

    pred = w.T @ pts
    return np.argmax(pred, axis=0)

def percent_correct(true_labels, hypothesis_labels):
    """
    Returns the percentage of labels that are correctly predicted.
    """
    return sum(true_labels == hypothesis_labels) / true_labels.size

def normalize_weights(w, d, k):
    """
    Takes in w, a np vector of size d * k, and returns the W matrix
    of size d by k, with columns normalized
    """
    w = w.reshape(d, k)
    return normalize(w, axis=0)

start = time.time()

lmbda = tune(train_pts, train_labels)
w_hat, loss_val = multiclass_svm(lmbda, train_pts, train_labels)
pred = predict(w_hat, test_pts)
print('percent correct: ' + str(percent_correct(test_labels, pred)))

end = time.time()
print('sec to run code: ' + str(end - start))

# Plot true weight vectors and guess
guess = normalize_weights(w_hat, d, k)
plt.figure()
ax = plt.gca()
X = np.zeros(k)
Y = np.zeros(k)
q1 = ax.quiver(X, Y, true[0], true[1], angles='xy', scale_units='xy', scale=1)
q2 = ax.quiver(X, Y, guess[0], guess[1], angles='xy', scale_units='xy', scale=1, color='rebeccapurple')
xmin = min(min(true[0]), min(guess[0]))
xmax = max(max(true[0]), max(guess[0]))
ymin = min(min(true[1]), min(guess[1]))
ymax = max(max(true[1]), max(guess[1]))
ax.set_xlim([xmin - 1, xmax + 1])
ax.set_ylim([ymin - 1, ymax + 1])
plt.title('Multiclass SVM')
plt.legend((q1, q2), ('True weights', 'Guess weights, normalized'))
plt.draw()
plt.show()


