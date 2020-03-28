import pandas as pd
# import sys
# import tokenize
import numpy as np
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# from nltk.tokenize import  porter
# nltk.download()
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------------------------------

quora_df = pd.read_csv(r"C:\Users\SANKET\PycharmProjects\Quora-challenge\train.csv",
                         engine='python')
quora_df.dropna(axis=0, inplace=True)
rowcount = len(quora_df.axes[0])

# ------------------------------------------------------------------------------------------------------
# Extracting useful columns
indexlist = []
train = round(0.70 * rowcount)

# extract information of datatypes of the columns
filteredColumns = quora_df.dtypes[quora_df.dtypes == np.object]
# get object datatype columns
indexlist = list(filteredColumns.index)

questions1 = []
questions2 = []

questions1 = quora_df[indexlist[0]].values.tolist()
questions2 = quora_df[indexlist[1]].values.tolist()
output = quora_df.iloc[:, -1].values.tolist()

questions1 = questions1[train:]
questions2 = questions2[train:]

train = rowcount - train

# --------------------Preprocessing-------------------------------------------------------------------------------------


def preprocessing(String, stopwordFlag=True,
                  stemFlag=True):  # default value is always true for stemming and stopwords

    token = nltk.word_tokenize(String)
    if stopwordFlag == False and stemFlag == False:
        token_string = " ".join(token)
        return token_string

    stopwords_list = set(stopwords.words('english'))
    wordsFiltered = []
    stemmedwords = []
    for w in token:
        if w not in stopwords_list:
            wordsFiltered.append(w)
    if stemFlag == False:
        stopwords_string = " ".join(stemmedwords)
        return stopwords_string

    ps = PorterStemmer()

    if stopwordFlag == False:
        for w in token:
            stemmedwords.append(ps.stem(w))
        stemwords_string = " ".join(stemmedwords)
        return stemwords_string
    for w in wordsFiltered:
        stemmedwords.append(ps.stem(w))
    stemwords_string = " ".join(stemmedwords)
    return stemwords_string


# ------------------Vectorization---------------------------------------------------------------------------------------

def vectorize(questions):
    tfidf = []
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    quetokens_df = pd.DataFrame(questions)
    tfidf = tfidf_vectorizer.fit_transform(quetokens_df[0])
    question_vector = tfidf.todense()
    return question_vector


# ------------------Similarity function---------------------------------------------------------------------------------


def Similarity(question1, question2):
    score = cosine_similarity(question1, question2)
    return score
# -----------------Evaluation-------------------------------------------------------------------------------------------
def evaluate(Similarity, target, threshold, train):
    count = 0
    predicted = []
    for i in range(0, train):
        if Similarity[i] > threshold:
            predicted.append(1)
        else:
            predicted.append(0)
    # for j in range (0,train):
    #     if predicted[j]==target[j]:
    #         #print("Yes")
    #         count=count+1
    # print(predicted[:6],target[:6])
    # print(count)
    # accuracy = float(count/train)
    #    print (predicted,target[:train])
    accuracy = accuracy_score(target[:train], predicted)
    return accuracy


# ----------------------------------------------------------------------------------------------------

# main
que1tokens = []
que2tokens = []
similarityscore = []
for i in range(0, train):
    que1tokens.append(preprocessing(questions1[i]))
    que2tokens.append(preprocessing(questions2[i]))

for i in range(0, train):
    combining = que1tokens.append(que2tokens[i])

VectorQue1 = vectorize(que1tokens)
# VectorQue2 = vectorization(que2tokens)

for i in range(0, train):
    similarityscore.append(Similarity(VectorQue1[train + i, :], VectorQue1[i, :]))

print(evaluate(similarityscore, output[:train], 0.90, train))

















