from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image
import pydotplus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Result of Real New is 0 and Fake news is 1

def load_data():
    '''

    return a clean txt file in numpy array in order of
    x_train_vector, x_validation_vector, x_test_vector, y_train_vector , y_validation_vector, y_test_vector
    
    :return: np.array

    '''
    file_real = open("clean_real.txt", "r").read()
    file_fake = open("clean_fake.txt", "r").read()
    real = file_real.split("\n")
    fake = file_fake.split("\n")
    real.pop()
    fake.pop()
    data = real + fake
    # Set Real new as 0 and fake news as 1
    label = [0] * len(real) + [1] * len(fake)
    vectorizer = CountVectorizer()
    data_vector = vectorizer.fit_transform(data)
    # Split data to Train and test
    x_train, x_test, y_train, y_test = train_test_split(data_vector, label, test_size=0.3, shuffle=True)
    # Split data to test and validation
    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, shuffle=True)
    x_train_array = x_train.toarray()
    y_train_array = np.asarray(y_train)
    x_validation_array = x_validation.toarray()
    y_validation_array = np.asarray(y_validation)
    x_test_array = x_test.toarray()
    y_test_array = np.asarray(y_validation)
    return x_train_array, x_validation_array, x_test_array, y_train_array, y_validation_array, y_test_array

def test_accuracy(model, x, y):
    '''
    return the accuracy of model
    :return : float
    '''
    res = model.predict(x)
    count = 0
    for i in range(len(res)):
        if res[i] == y[i]:
            count += 1
    return count/len(res)



def select_tree_model():
    '''
    return the model with best accuarcy 
    :return: DecisionTreeClassifier()
    '''
    x_train_array, x_validation_array, x_test_array, y_train_array, y_validation_array, y_test_array = load_data()
    # Set up models use Gini with 5 different max_depth
    gini10 = DecisionTreeClassifier(criterion="gini", max_depth=10)
    gini30 = DecisionTreeClassifier(criterion="gini", max_depth=30)
    gini50 = DecisionTreeClassifier(criterion="gini", max_depth=50)
    gini100 = DecisionTreeClassifier(criterion="gini", max_depth=100)
    gini150 = DecisionTreeClassifier(criterion="gini", max_depth=150)
    gini10.fit(x_train_array, y_train_array)
    gini30.fit(x_train_array, y_train_array)
    gini50.fit(x_train_array, y_train_array)
    gini100.fit(x_train_array, y_train_array)
    gini150.fit(x_train_array, y_train_array)
    # Set up models use Entropy with 5 different max_depth
    entro10 = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    entro30 = DecisionTreeClassifier(criterion="entropy", max_depth=30)
    entro50 = DecisionTreeClassifier(criterion="entropy", max_depth=50)
    entro100 = DecisionTreeClassifier(criterion="entropy", max_depth=100)
    entro150 = DecisionTreeClassifier(criterion="entropy", max_depth=150)
    entro10.fit(x_train_array, y_train_array)
    entro30.fit(x_train_array, y_train_array)
    entro50.fit(x_train_array, y_train_array)
    entro100.fit(x_train_array, y_train_array)
    entro150.fit(x_train_array, y_train_array)
    models = [gini10, gini30, gini50, gini100, gini150, entro10, entro30, entro50, entro100, entro150]
    scores = []
    # get models accuarcy on validation set
    for model in models:
        score = model.score(x_validation_array, y_validation_array)
        scores.append(score)
        print(model.criterion, model.max_depth, 'Model Accuarcy:', score)
    best = max(scores)
    best_model = models[scores.index(best)]
    print('Best Model :', best_model)
    print("Best Model Accuracy ", test_accuracy(best_model, x_test_array, y_test_array))
    return best_model



def export_tree_diagram(model):
    '''
    Return the diagram for question 3c
    '''
    # Only works in Jupyter Notebook
    dot_data = StringIO()
    export_graphviz(model,max_depth=2, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('tree.png')
    Image(graph.create_png())



def entropy(p:float):
    '''
    return entropy of p
    '''
    if p == 0:
        return 0
    return (-p*math.log2(p)-(1-p)*math.log2(1-p))



def compute_info_gain(x, y, keyword):
    '''
    return information gain of the split
    '''
    real_count = 0
    real_appear = 0
    fake_appear = 0
    real_not = 0
    fake_not = 0
    for i in range(len(y)):
        res = y[i]
        if res == 0:
            real_count += 1
        if keyword in x[i]:
            if res == 0:
                real_appear += 1
            else:
                fake_appear += 1
        else:
            if res == 0:
                real_not += 1
            else:
                fake_not += 1
    appear = real_appear + fake_appear
    not_appear = real_not + fake_not
    fake = len(y) - real_count
    #H(Y=0)
    p_real = (real_appear+real_not)/len(x)
    #H(Y=1)
    # H(Y)
    root_entro = entropy(p_real) 
    # H(Y|keyword appear) + H(Y|keyword not appear)
    appear_entro = entropy(real_appear/appear)
    not_appear_entro = entropy(real_not/not_appear)
    info = root_entro - (appear/len(x))*appear_entro - (not_appear/len(x))*not_appear_entro
    return info




def select_knn_model():
    '''
    return the knn model with best accuarcy 
    :return: KNeighborsClassifier()
    '''
    x_train_array, x_validation_array, x_test_array, y_train_array, y_validation_array, y_test_array = load_data()
    knn_models = []
    # Initialize models from range 1 to 20
    for i in range(1, 21):
        locals()['knn_' + str(i)] = KNeighborsClassifier(n_neighbors=i)
        knn_models.append(locals()['knn_' + str(i)])
    for model in knn_models:
        model.fit(x_train_array, y_train_array)
    knn_scores = []
    # get models accuracy on validation set
    for model in knn_models:
        score = model.score(x_validation_array, y_validation_array)
        knn_scores.append(score)
    knn_best = max(knn_scores)
    best_knn_model = knn_models[knn_scores.index(knn_best)]
    knn_test_scores = []
    # get model scores on test set for plot
    for model in knn_models:
        score = model.score(x_test_array, y_test_array)
        knn_test_scores.append(score)
    # Plot ---------------
    plt.plot(pd.Series(range(1,21)), pd.Series(knn_scores), marker='o', linewidth=1, alpha=0.9, label="validation")
    plt.plot(pd.Series(range(1,21)), pd.Series(knn_test_scores), marker='o', color = 'red', linewidth=1, alpha=0.9, label="test")
    plt.xlabel("k - number of neighbors")
    plt.xticks([2,4,6,8,10,12,14,16,18,20])
    plt.ylabel("Accuarcy")
    plt.legend()
    # Plot end-------------
    score = best_knn_model.score(x_test_array, y_test_array)
    print("Best Model: ", best_knn_model)
    print("Best model Accuarcy on test set: ", score)
    return best_knn_model



