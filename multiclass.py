from scipy.optimize import minimize
import numpy as np
import numpy.random as random
import time

random.seed(37)

# Verify example from 17.2
d = 2
k = 4
true = random.randn(d, k)

m = 420
train_pts = random.randn(d, m)
train_labels = true.T @ train_pts
train_labels = np.argmax(train_labels, axis=0)

m_test = int(0.2 * m)
test_pts = random.randn(d, m_test)
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
            delta = np.ones(k)
            delta[labels[j]] -= 1
            pt_loss = np.max(delta + w.T @ data[:, j]) - w[:, labels[j]] @ data[:, j]
            loss += (1 / m) * pt_loss
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

start = time.time()

lmbda = tune(train_pts, train_labels)
w_hat, loss_val = multiclass_svm(lmbda, train_pts, train_labels)
pred = predict(w_hat, test_pts)
print("percent correct: " + str(percent_correct(test_labels, pred)))

end = time.time()
print("sec to run code: " + str(end - start))
