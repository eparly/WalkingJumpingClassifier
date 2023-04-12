from featext import *
from prepro import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

# instantiante regression and clf
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)

# fit into classifier
clf.fit(X_train, y_train)

# obtain predictions and probabilities
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)

# accuracy & recall
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# confusion matrix

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
# cm_display.show()

# ROC
fpr, tpr = roc_curve(y_test, y_clf_prob[:, 1], pos_label = clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# roc_display.show()

# AUC

auc = roc_auc_score(y_test, y_clf_prob[:,1])