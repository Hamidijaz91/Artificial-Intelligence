import nltk
nltk.download('webtext')
nltk.download('stopwords')

from nltk.tokenize import PunktSentenceTokenizer as PST
from nltk.corpus import webtext
from nltk.tokenize import sent_tokenize as st
text = webtext.raw('D:/PyCharmProjects/Lab 5 Tokenization and StopWords/train_tokenize.txt')
senttokenize = PST(text)
sentence = senttokenize.tokenize(text)
print(sentence)
print(sentence[-2])
# New Program From HEre
# StopWords removing
from nltk.corpus import stopwords
englishStopwords = set(stopwords.words('english'))
words = ['i','am', 'a' ,'writer']
result = [words for words in words if words not in englishStopwords]
print(result)
print(stopwords.fileids())
