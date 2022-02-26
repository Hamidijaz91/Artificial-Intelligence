import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import sent_tokenize as st

sentence = "This is 2k19. Semester 6th"

WordToken = wt(sentence)
SentToken = st(sentence)
print(WordToken)
print(SentToken)




