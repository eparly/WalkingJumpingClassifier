from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import pickle
import matplotlib.pyplot as plt

def runmodel(train_features, y_train, test_features, y_test):

    # instantiate regression and clf
    l_reg = LogisticRegression(max_iter=10000)
    clf = make_pipeline(StandardScaler(), l_reg)

    # fit into classifier
    clf.fit(train_features, y_train)

    # obtain predictions and probabilities
    y_pred = clf.predict(test_features)
    y_clf_prob = clf.predict_proba(test_features)

    # accuracy & recall
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()

    # ROC 
    fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label = clf.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()

    # AUC

    auc = roc_auc_score(y_test, y_clf_prob[:,1])
    pickle.dump(clf, open("model.pkl", "wb"))

#make a prediction
def predict(Features):
    clf = pickle.load(open("model.pkl", "rb"))
    y_pred = clf.predict(Features)

    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            print("Window ", i, ": Jumping")
        else:
            print("Window ", i, ": Walking")
    
    return y_pred
