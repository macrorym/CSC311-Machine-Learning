'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for i in range(0, 10):
        ind = np.where(train_labels == i)
        digits = train_data[ind]
        means[i] = np.mean(digits, axis=0)
    # Compute means
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    # Compute covariances
    for i in range(0, 10):
        ind = np.where(train_labels == i)
        digit = train_data[ind]
        mu = np.mean(digit, axis=0)
        va = digit - mu
        for k in range(64):
            for j in range(64):
                 covariances[i][j,k]= np.dot(va[:,k], va[:,j].transpose())/(len(digit))
    return covariances


def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # ...
        log_cov = np.log(cov_diag.reshape((8,8)))
        cov.append(log_cov)
    all_concat = np.concatenate(cov, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    res = []
    for digit in digits:
        row = []
        for i in range(0, 10):
            covs = covariances[i]
            mus = means[i]
            x_mu = digit - mus
            p = -0.5 * (np.log(np.linalg.det(covs)) + (x_mu).dot(np.linalg.inv(covs)).dot(x_mu.T) + 64 * np.log(
                2 * np.pi))
            row.append(p)
        res.append(row)
    return np.array(res)


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    glk = generative_likelihood(digits, means, covariances)
    pxypy = glk + np.log(0.1)
    px = logsumexp(pxypy, axis=1).reshape(-1, 1)
    return pxypy-px


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    p = 0
    N = len(cond_likelihood)
    for i in range(N):
        k = int(labels[i])
        p += cond_likelihood[i,k]
    return p/N



def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    like = np.exp(cond_likelihood)
    res = []
    for i in range(len(cond_likelihood)):
        lst = like[i].tolist()
        res.append(lst.index(max(lst)))
    # Compute as described above and return
    return np.array(res)

def model_accuracy(digits, labels, means, covariances):
    '''
    report model accuracy on given data
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    like = np.exp(cond_likelihood)
    count = 0
    for i in range(len(like)):
        lst = like[i].tolist()
        if  lst.index(max(lst)) == labels[i]:
            count += 1
    return count/len(like)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    plot_cov_diagonal(covariances)
    # Evaluation
    print("Average Conditional Log-Likelihood")
    print('Training Set:', avg_conditional_likelihood(train_data, train_labels, means, covariances))
    print('Test Set', avg_conditional_likelihood(test_data, test_labels, means, covariances))

    print('\n \n Accuracy of Model:')
    print('Training Set:', model_accuracy(train_data, train_labels, means, covariances))
    print('Test Set', model_accuracy(test_data, test_labels, means, covariances))
if __name__ == '__main__':
    main()
