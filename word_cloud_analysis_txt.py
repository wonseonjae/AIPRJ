from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from konlpy.tag import Hannanum

import re

myHannanum = Hannanum()

text = open("contents.txt", encoding="UTF-8").read()

replace_text = re.sub("[!@$#%^&*()_+]", " ", text)

analysis_text = (" ".join(myHannanum.nouns(replace_text)))

stopwords = set(STOPWORDS)
stopwords.add("을")
stopwords.add("를")
stopwords.add("있는")
stopwords.add("역할을")
stopwords.add("대구에")

myWC = WordCloud(font_path="font/NanumBrush.ttf", stopwords=stopwords, background_color="skyblue").generate(analysis_text)

plt.figure(figsize=(5, 5))

plt.imshow(myWC, interpolation="lanczos")

plt.axis("off")

plt.show()