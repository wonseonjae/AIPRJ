from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
import re

text = open("contents.txt", encoding="UTF-8").read()

okt = Okt()

token = re.sub("(\.)", "", text)

token = okt.morphs(token)

corpus = [" ".join(token)]

vector = CountVectorizer()

print(vector.fit_transform(corpus).toarray())

print(vector.vocabulary_)
