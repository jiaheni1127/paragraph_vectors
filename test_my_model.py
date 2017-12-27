# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

model = Doc2Vec.load('./imdb.d2v')

# print(model.docvecs["TRAIN_POS_1"])
#----------------------------------------------------
# train_arrays = numpy.zeros((25000, 100))
train_arrays = numpy.zeros((25000, 200))
train_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_train_pos = "TRAIN_POS_" + str(i)
    prefix_train_neg = "TRAIN_NEG_" + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0
# ----------------------------------------------------
# test_arrays = numpy.zeros((25000, 100))
test_arrays = numpy.zeros((25000, 200))
test_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_test_pos = "TEST_POS_" + str(i)
    prefix_test_neg = "TEST_NEG_" + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[12500 + i] = 0
# ----------------------------------------------------
classifier = LogisticRegression()
# classifier = SVC()
# classifier = GaussianNB()
# classifier = KNeighborsClassifier()
# classifier = DecisionTreeClassifier()
classifier.fit(train_arrays, train_labels)
print(classifier.score(test_arrays, test_labels))
