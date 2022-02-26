import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
dataset = pd.read_csv('ds.csv')

X = dataset.iloc[:,[0,1]].values
y = dataset.iloc[:,[2]].values

X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.6, random_state = 0 )

model=LR()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


print(y_pred)