'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import collections


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        dis = self.l2_distance(test_point)
        digit = self.find_digit(dis, k)     
        return digit

    def find_digit(self, dis, k):
        '''
        You should return the digit label provided by the algorithm
        '''
        sorted_dis = np.sort(dis, axis=None)
        lst = dis.tolist()
        k_nei = sorted_dis[:k]
        result = []
        for nei in k_nei:
          i = lst.index(nei)
          result.append(self.train_labels[i])
        # Find the highest frequency digit
        coll = collections.Counter(result).most_common()
        if len(coll) == 1:
          digit = result[0]
        else:
          # break ties
          if coll[0][1] == coll[1][1]:
            # increase k value by 1 
            digit = self.find_digit(dis, k+1)
          else:
            digit = collections.Counter(result).most_common()[0][0]       
        return digit

    def pred_probs(self, test_point, k):
        '''
        Return Probability of test point with given k
        '''
        probs = []
        for point in test_point:
            dis = self.l2_distance(point)
            sorted_dis = np.sort(dis, axis=None)
            lst = dis.tolist()
            k_nei = sorted_dis[:k]
            result = []
            for nei in k_nei:
              i = lst.index(nei)
              result.append(self.train_labels[i])
            coll = collections.Counter(result).items()
            prob = np.zeros(10)
            for tup in coll:
                prob[int(tup[0])] = tup[1]/k
            probs.append(prob)
        return np.array(probs)
    
    def predict(self, test_points, k):
        pred = []
        for point in test_points:
            pred.append(self.query_knn(point, k))
        return np.array(pred)
            
            
def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    k_acc = []
    for k in k_range:
      # Loop over folds
      r = len(train_data) // 10
      folds = []
      accus = []
      for i in range(10):
        x_test = train_data[i * r:r * (i + 1)]
        y_test = train_labels[i * r:r * (i + 1)]
        x_train = np.concatenate((train_data[:i * r], train_data[r * (i + 1):]), 0)
        y_train = np.concatenate((train_labels[:i * r], train_labels[r * (i + 1):]), 0)
        knn = KNearestNeighbor(x_train, y_train)
        accu = classification_accuracy(knn, k, x_test, y_test)
        accus.append(accu)
        # Evaluate k-NN
      avg_acc = sum(accus)/10
      k_acc.append(avg_acc)
    best_k = k_acc.index(max(k_acc)) + 1
    print("Optimal K: ", best_k)
    print("Average Accuracy across folds: ", max(k_acc))
    return best_k

    

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    y_hats = []
    for test_point in eval_data:
      y_hats.append(knn.query_knn(test_point, k))
    count = 0
    for i in range(len(y_hats)):
      if y_hats[i] == eval_labels[i]:
        count += 1
    accuracy = count/len(y_hats)
    return accuracy



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    predicted_label = knn.query_knn(test_data[0], 1)
    print('K = 1 Train Accuarcy: ', classification_accuracy(knn, 1, train_data, train_labels))
    print('K = 1 Test Accuarcy: ', classification_accuracy(knn, 1, test_data, test_labels))

    print('K = 15 Train Accuarcy: ', classification_accuracy(knn, 15, train_data, train_labels))
    print('K = 15 Test Accuarcy: ', classification_accuracy(knn, 15, test_data, test_labels))   

    opt_k = cross_validation(train_data, train_labels)
    print('Test Accuracy: ', classification_accuracy(knn, opt_k, test_data, test_labels))
if __name__ == '__main__':
    main()
