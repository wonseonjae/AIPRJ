import pandas as pd

from math import log

from konlpy.tag import Hannanum

import re

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc

    return log(N/(df+1))

def tfidf(t, d):
    return tf(t, d)* idf(t)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width', None)

myHannanum = Hannanum()

org_docs = [
    "학생들은 빅데이터와 인공지능 기술을 배우고 있다.",
    "빅데이터 기술은 방대한 데이터를 처리한다. 빅데이터는 많은 데이터를 저장한다.",
    "빅데이터 기술은 많이 어렵다. 특히 하둡이 어렵다.",
    "나의 목표는 빅데이터 기술을 활용하는 빅데이터 소프트웨어 개발자이다.",
    "소프트웨어 개발은 코딩이 필수이다. 나는 소프트웨어 개발자가 되고 싶다. 소프트웨어 개발자 화이팅!",
    "인공지능 기술에서 자연어 처리는 재밌다. 자연어는 사람이 사용하는 일반적인 언어이다."
]

docs = []

for org_doc in org_docs:
    replace_doc = re.sub("[!@#$%^&*()_+]", " ", org_doc)
    docs.append(" ".join(myHannanum.nouns(replace_doc)))

print(docs)

vocab = list(set(w for doc in docs for w in doc.split()))

vocab.sort()

print("중복 제거 단어 : " + str(vocab))

N = len(docs)

print("문서의 수 : " + str(N))

result = []

for i in range(N):
    result.append([])
    d = docs[i]

    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = vocab)

print("TF 결과")
print(tf_)

result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index = vocab, columns=["IDF"])

print("IDF 결과")
print(idf_)

result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d))

tfidf_ = pd.DataFrame(result, columns=vocab)


print("TF-IDF 결과")
print(tfidf_)

