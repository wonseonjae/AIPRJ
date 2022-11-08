from konlpy.tag import Okt, Hannanum
import re

myHannanum = Hannanum()

text = open("contents.txt", encoding="UTF-8").read()

analysis_text = (" ".join(myHannanum.nouns(text)))

okt = Okt()

token = re.sub("(\.)", "", analysis_text)

token=okt.morphs(token)

word2index={}
bow=[]

for word in token:

    if word not in word2index.keys():

        word2index[word] = len(word2index)

        bow.insert(len(word2index)-1, 1)

    else:
        index = word2index.get(word)

        bow[index] = bow[index]+1
print(word2index)

print(bow)
