'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    for i in range(10):
        ind = np.where(train_labels == i)
        digit = train_data[ind]
        njk = np.mean(digit, axis=0)
        eta[i] = njk
    return eta


def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    images = []
    for i in range(10):
        img = class_images[i]
        images.append(img.reshape((8,8)))
    all_concat = np.concatenate(images, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for i in range(10):
        for k in range(64):
            njk = eta[i,k]
            bj = np.random.beta(2,2)
            b = np.random.binomial(n=1, p=njk, size=1)
            generated_data[i,k] = bj*b
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    res = []
    for i in range(len(bin_digits)):
        b = bin_digits[i]
        row = []
        for k in range(10):
            p_b = 1
            for j in range(64):
                nkj = eta[k][j]
                bj = b[j]
                p_b = p_b*(((nkj)**(bj)) * ((1-nkj)**(1-bj)))
            row.append(p_b)
        res.append(row)
    return np.log(np.array(res))

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    px_y = generative_likelihood(bin_digits, eta)
    pxypy = px_y + np.log(0.1)
    return pxypy

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    p = 0
    N = len(cond_likelihood)
    for i in range(N):
        k = int(labels[i])
        p += cond_likelihood[i,k]
    return p/N


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    like = np.exp(cond_likelihood)
    res = []
    for i in range(len(cond_likelihood)):
        lst = like[i].tolist()
        res.append(lst.index(max(lst)))
    # Compute as described above and return
    return np.array(res)

def model_accuracy(bin_digits, labels, eta):
    '''
    report model accuracy on given data
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    like = np.exp(cond_likelihood)
    count = 0
    for i in range(len(like)):
        lst = like[i].tolist()
        if  lst.index(max(lst)) == labels[i]:
            count += 1
    return count/len(like)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)

    generate_new_data(eta)

    print("Average Conditional Log-Likelihood")
    print('Training Set:', avg_conditional_likelihood(train_data, train_labels, eta))
    print('Test Set', avg_conditional_likelihood(test_data, test_labels, eta))

    print('\n \n Accuracy of Model:')
    print('Training Set:', model_accuracy(train_data, train_labels, eta))
    print('Test Set', model_accuracy(test_data, test_labels, eta))

if __name__ == '__main__':
    main()
