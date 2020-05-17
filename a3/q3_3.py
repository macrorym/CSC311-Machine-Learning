from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, label_binarize
from sklearn.neural_network import MLPClassifier
from scipy.special import softmax
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc#, plot_confusion_matrix, RocCurveDisplay, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import matplotlib.pyplot as plt
# This may raise ValueErorr for old version of scikit-learn
from sklearn.metrics import plot_confusion_matrix
from data import load_all_data
from q3_1 import KNearestNeighbor



train_data, train_labels, test_data, test_labels = load_all_data('data')

enco = OneHotEncoder()
train_labels_o = enco.fit_transform(train_labels.reshape((len(train_labels),1))).toarray()
test_labels_o = enco.fit_transform(test_labels.reshape((len(test_labels),1))).toarray()

#KNN
knn = KNearestNeighbor(train_data, train_labels)


#MLP 
MLP = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation = 'relu', solver='adam')
MLP.fit(train_data, train_labels_o)


#SVM
svm = LinearSVC()
svm.fit(train_data, train_labels)


#Adaboost Classifier
base = DecisionTreeClassifier(max_depth=1)
'''
res = []
for i in range(1, 101):
    ada = AdaBoostClassifier(base_estimator = base, n_estimators = i)
    ada.fit(train_data, train_labels)
    res.append(ada.score(test_data, test_labels))
res.index(max(res))
>>> 41
'''
ada = AdaBoostClassifier(base_estimator = base, n_estimators = 41)
ada.fit(train_data, train_labels)



def conf_m(model):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    plot_confusion_matrix(model, test_data, test_labels,
                          cmap=plt.cm.Blues, normalize = 'true', ax=ax)


def conf_mlp(model):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    plot_confusion_matrix(model, test_data, enco.transform(test_labels).reshape(1,len(test_labels)),
                          cmap=plt.cm.Blues, normalize = 'true', ax=ax)


def precision_recall(conf):
    recall = []
    prec = []
    for i in range(len(conf)):
        tp = conf[i][i]
        prec.append(tp/sum(conf[:, i]))
        recall.append(tp/sum(conf[i, :]))
    return prec, recall


def accuracy(conf):
    to = 0
    cor = 0
    for i in range(len(conf)):
        to += sum(conf[i])
        cor += conf[i][i]
    return cor/to


def model_info(y_h):
    conf = confusion_matrix(test_labels, y_h)
    prec, recall = precision_recall(conf)
    d = {'Precision': prec, 'Recall': recall}
    df = pd.DataFrame(data = d)
    print(df)
    print('Accuracy: ', accuracy(conf))


def plot_knn_roc(model):
    probs = model.pred_probs(test_data, 1)
    fig, ((ax0,ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11))= plt.subplots(4,3, figsize=(16, 16))
    for i in range(10):
        prob = probs[:,i]
        fpr, tpr, thresholds = roc_curve(test_labels_o[:, i], prob)
        roc_auc = auc(fpr, tpr)
        label = 'KNN AUC:' + ' {0:.2f}'.format(roc_auc)
        locals()['ax'+str(i)].plot(fpr, tpr, c = 'b', label = label, linewidth = 2)
        locals()['ax'+str(i)].set_xlabel('False Positive Rate', fontsize = 12)
        locals()['ax'+str(i)].set_ylabel('True Positive Rate', fontsize = 12)
        locals()['ax'+str(i)].set_title('ROC' + "  class {}".format(i), fontsize = 12)
        locals()['ax'+str(i)].legend(loc = 'lower right', fontsize = 12)
    fig.tight_layout(pad=3.0)
    fig.savefig('knn_roc.png')
    fig.show()



def plot_mlp_roc(model):
    probs = model.predict_proba(test_data)
    fig, ((ax0,ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11))= plt.subplots(4,3, figsize=(16, 16))
    for i in range(10):
        prob = probs[:,i]
        fpr, tpr, thresholds = roc_curve(test_labels_o[:, i], prob)
        roc_auc = auc(fpr, tpr)
        label = 'MLP AUC:' + ' {0:.2f}'.format(roc_auc)
        locals()['ax'+str(i)].plot(fpr, tpr, c = 'b', label = label, linewidth = 2)
        locals()['ax'+str(i)].set_xlabel('False Positive Rate', fontsize = 12)
        locals()['ax'+str(i)].set_ylabel('True Positive Rate', fontsize = 12)
        locals()['ax'+str(i)].set_title('ROC' + "  class {}".format(i), fontsize = 12)
        locals()['ax'+str(i)].legend(loc = 'lower right', fontsize = 12)
    fig.tight_layout(pad=3.0)
    fig.savefig('mlp_roc.png')
    fig.show()

    
def plot_svm_roc(model):
    probs = model.decision_function(test_data)
    fig, ((ax0,ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11))= plt.subplots(4,3, figsize=(16, 16))
    for i in range(10):
        prob = probs[:,i]
        fpr, tpr, thresholds = roc_curve(test_labels_o[:, i], prob)
        roc_auc = auc(fpr, tpr)
        label = 'SVM AUC:' + ' {0:.2f}'.format(roc_auc)
        locals()['ax'+str(i)].plot(fpr, tpr, c = 'b', label = label, linewidth = 2)
        locals()['ax'+str(i)].set_xlabel('False Positive Rate', fontsize = 12)
        locals()['ax'+str(i)].set_ylabel('True Positive Rate', fontsize = 12)
        locals()['ax'+str(i)].set_title('ROC' + "  class {}".format(i), fontsize = 12)
        locals()['ax'+str(i)].legend(loc = 'lower right', fontsize = 12)
    fig.tight_layout(pad=3.0)
    fig.savefig('svm_roc.png')
    fig.show()



def plot_ada_roc(model):
    probs = model.predict_proba(test_data)
    fig, ((ax0,ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11))= plt.subplots(4,3, figsize=(16, 16))
    for i in range(10):
        prob = probs[:,i]
        fpr, tpr, thresholds = roc_curve(test_labels_o[:, i], prob)
        roc_auc = auc(fpr, tpr)
        label = 'ADA AUC:' + ' {0:.2f}'.format(roc_auc)
        locals()['ax'+str(i)].plot(fpr, tpr, c = 'b', label = label, linewidth = 2)
        locals()['ax'+str(i)].set_xlabel('False Positive Rate', fontsize = 12)
        locals()['ax'+str(i)].set_ylabel('True Positive Rate', fontsize = 12)
        locals()['ax'+str(i)].set_title('ROC' + "  class {}".format(i), fontsize = 12)
        locals()['ax'+str(i)].legend(loc = 'lower right', fontsize = 12)
    fig.tight_layout(pad=3.0)
    fig.savefig('ada_roc.png')
    fig.show()


if __name__ == '__main__':
    print("\n \n----KNN----")
    knn_h = knn.predict(test_data, 1)
    model_info(knn_h)
    plot_knn_roc(knn)
    print("\n \n----MLP----")
    mlp_h = enco.inverse_transform(MLP.predict(test_data))
    model_info(mlp_h)
    plot_mlp_roc(MLP)
    print("\n \n----SVM----")
    svm_h = svm.predict(test_data)
    model_info(svm_h)
    plot_svm_roc(svm)
    print("\n \n ----AdaBoost----")
    ada_h = ada.predict(test_data)
    model_info(ada_h)
    plot_ada_roc(ada)
