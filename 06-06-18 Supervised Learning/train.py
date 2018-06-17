import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from nltk.tokenize import RegexpTokenizer

from stw import SupervisedTermWeightingWTransformer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold





#Membaca file data training dan data testing
training = pd.read_csv('data_training2.csv' ,header = 0)



#Menunjukkan kolom sentimen dan tweet dari Data Training
training.head()
trainx = training["filename"]
trainy = training["target"]
#trainx = trainx.str.lower()
#trainx = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','', trainx)
#print(trainx)

# Construct term count matrix for train and test datasets
vectorizer = CountVectorizer(ngram_range=(1,3))
analyzer = vectorizer.build_analyzer()

train_x, test_x, train_y, test_y = train_test_split(trainx, trainy, test_size=0.2, random_state=0)

#Pengubahan dan perhitungan term frequency menjadi bentuk vektor
#train_x = vectorizer.fit_transform(training['filename'])
#train_y = training['target']
trainx_cv = vectorizer.fit_transform(train_x.values.astype('U'))
#x_array = trainx_cv.toarray()
#print ((x_array))

#test_x = vectorizer.transform(testing['filename'])
#test_y = testing['target']
testx_cv = vectorizer.transform(test_x.values.astype('U'))




# Inisiasi SVM Classifier

clf = LinearSVC()


# Pembobotan TF-IDF


transformer = TfidfTransformer()

train_x_t = transformer.fit_transform(trainx_cv,train_y) #kalau pakai teks split pakai trainx_cv
test_x_t  = transformer.transform(testx_cv) #testx_cv
#print(train_x_t.toarray())


# Train classifier dan membuat prediksi

clf.fit(train_x_t,train_y)
pred = clf.predict(test_x_t)


# Menyimpan model klasifikasi dengan Pickle

svm_clf_pkl_filename = 'svm_clf.pickle' #membuat file untuk menyimpan model klasifikasi
svm_clf_model_pkl = open(svm_clf_pkl_filename, 'wb') #membuka file 
pickle.dump(clf,svm_clf_model_pkl) #menyimpan model pada file pickle
svm_clf_model_pkl.close() #menutup file pickle

#svm_clf_model_pkl = open(svm_clf_pkl_filename, 'rb')
#clf = pickle.load(svm_clf_model_pkl)
#print ('model',clf)

# print performa
print ('tf-idf scheme: accuracy = %f, recall = %f, f1 score = %f, precision = %f' % \
(accuracy_score(test_y,pred), recall_score(test_y,pred), f1_score(test_y,pred), precision_score(test_y,pred)))
print("confusion matrix :")
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))

# Memanggil Supervised term weighting schemes dari file stw.py 
for scheme in ['tfchi2','tfor','tfrf']:

    transformer = SupervisedTermWeightingWTransformer(scheme=scheme)

    train_x_t = transformer.fit_transform(trainx_cv,train_y)
    test_x_t  = transformer.transform(testx_cv)
    #print (train_x_t.toarray())

    # Train classifier dan membuat prediksi
    clf.fit(train_x_t,train_y)
    pred = clf.predict(test_x_t)


    # print performa
    print ('%s scheme: accuracy = %f, recall = %f, f1 score = %f, precision = %f, confusion_matrix =%s, classification_report=%s' % \
    (scheme, accuracy_score(test_y,pred), recall_score(test_y,pred), f1_score(test_y,pred), precision_score(test_y,pred), confusion_matrix(test_y,pred),classification_report(test_y,pred)))
