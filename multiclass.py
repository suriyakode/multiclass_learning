from scipy.optimize import minimize
import numpy as np
import numpy.linalg as la
import numpy.random as random
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import time

def generate_points(d, k, m_train, m_test, dist):
    """
    Generate true weights, training points, and test points.
    d :         Ambient dimension
    k :         Number of classes
    m_train :   Number of points in training set
    m_test :    Number of points in test set

    Returns
    true :          k weight vectors in d dimensions as a d by k matrix
    train_pts :     m_train vectors in d dimensions as d by m_train matrix,
                    the columns of which are in the unit ball
    train_labels :  Labels for the train_pts as given by the true weights
    test_pts :      m_test vectors in d dimensions as a d by m_test matrix,
                    the columns of which are in the unit ball
    test_labels :   Labels for the test_pts as given by the true weights
    dist :          Distribution from which to generate weights and points
    """
    true = dist(d, k)
    true = normalize(true, axis=0)

    train_pts = dist(d, m_train)
    train_pts = train_pts / max(la.norm(train_pts, axis=0))
    # train_pts = normalize(train_pts, axis=0)
    train_labels = true.T @ train_pts
    train_labels = np.argmax(train_labels, axis=0)

    test_pts = dist(d, m_test)
    test_pts = test_pts / max(la.norm(test_pts, axis=0))
    # test_pts = normalize(test_pts, axis=0)
    test_labels = true.T @ test_pts
    test_labels = np.argmax(test_labels, axis=0)

    return true, train_pts, train_labels, test_pts, test_labels


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

def multiclass_svm(lmbda, pts, labels, dist=random.randn):
    loss_fn = svm_loss(lmbda, pts, labels)
    w_guess = dist(d * k)
    res = minimize(loss_fn, w_guess)
    loss_val = loss_fn(res.x)
    return res.x, loss_val

def tune(pts, labels, start=-5, end=5, num=11):
    """
    Use the provided pts and associated labels to determine the
    best lambda to run multiclass svm. Uses log spacing for
    lambdas to test against
    pts :       np array of size d * m where m is number of points
    labels :    np vector of size m
    start :     start of log space
    end :       end of log space
    num :       number of lambdas to try
    """
    # Hold out 25% of data to evaluate loss
    m = pts.shape[1]
    m_holdout = int(0.25 * m)
    holdout_pts = pts[:, -m_holdout:]
    holdout_labels = labels[-m_holdout:]

    pts = pts[:, :-m_holdout]
    labels = labels[:-m_holdout]

    lmbdas = np.logspace(start, end, num)
    best_loss = np.Infinity
    best_lmbda = -1
    for lmbda in lmbdas:
        w, _ = multiclass_svm(lmbda, pts, labels)
        loss = percent_correct(holdout_labels, predict(w, holdout_pts))
        if loss < best_loss:
            best_loss = loss
            best_lmbda = lmbda
    return best_lmbda

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

def plot_weights(true, w_hat, d, k):
    """
    Plot (in 2D) the true weight vectors and the guesses for them
    w_hat :     A numpy vector of length d * k
    true :      The true weights as a d by k matrix
    d :         The ambient dimension
    k :         The number of classes
    """
    assert d == 2
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

def plot_acc_vs_samples(d, k, m_test, m_min=10, m_max=500, num=25, dist=random.randn):
    accuracies = []
    samples = [int(m) for m in np.linspace(m_min, m_max, num)]
    true, full_train_pts, full_train_labels, \
            test_pts, test_labels = \
            generate_points(d, k, m_max, m_test, dist)
    for m in samples:
        # Pick first m samples
        train_pts = full_train_pts[:, :m]
        train_labels = full_train_labels[:m]

        # Train, predict, and add accuracy to list
        lmbda = 1e-2
        w_hat, _ = multiclass_svm(lmbda, train_pts, train_labels, dist)
        pred = predict(w_hat, test_pts)
        acc = percent_correct(test_labels, pred)
        accuracies.append(acc)
        print('m = {}, acc = {}'.format(m, acc))
    plt.title('Accuracy vs number of samples for d = {}'.format(d))
    plt.plot(samples, accuracies, '-')
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy on {} test points'.format(m_test))
    plt.show()
    return accuracies, samples

def unif(*dims):
    """
    Sample uniformly from [-1, 1]
    """
    sample = random.rand(*dims)
    return (sample * 2) - 1

# Visualize vectors in 2d
random.seed(37)
start = time.time()

d = 2
k = 4
true, train_pts, train_labels, test_pts, test_labels = generate_points(d, k, 420, int(100))

lmbda = tune(train_pts, train_labels)
w_hat, loss_val = multiclass_svm(lmbda, train_pts, train_labels)
pred = predict(w_hat, test_pts)
print('percent correct: ' + str(percent_correct(test_labels, pred)))

end = time.time()
print('sec to run code: ' + str(end - start))

# Plot true weight vectors and guess
plot_weights(true, w_hat, d, k)

# Plot accuracies vs samples with points generated from random normal dist
random.seed(37)
start = time.time()
accuracies, samples = plot_acc_vs_samples(d, k, 100, 10, 1000, 25)
end = time.time()
print('sec to run acc vs samples plots: ' + str(end - start))
print('accuracies = {}'.format(accuracies))
print('samples = {}'.format(samples))

# Plot accuracies vs samples with points generated from uniform dist
random.seed(37)
start = time.time()
accuracies, samples = plot_acc_vs_samples(d, k, 100, 10, 1000, 25, unif)
end = time.time()
print('sec to run acc vs samples plots: ' + str(end - start))
print('accuracies = {}'.format(accuracies))
print('samples = {}'.format(samples))

