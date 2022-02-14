
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR

#read the Dataset
dataset = pd.read_csv('dataset.csv')
print(dataset)

#distribute into dependent/independent
x = dataset.iloc[:,1].values
y = dataset.iloc[:,0:1].values

#slip the data into training/testing
x_train, x_test, y_train, y_test = tts(x,y, train_size=0.8, random_state=1)

#choose the model
model = LR()

#data preprocessing
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

#train the model
model.fit(x_train, y_train)

#test the model
y_pred = model.predict(x_test)
print(x_test,y_pred)
#data Visualization
plt.scatter(x_train, y_train, marker="<", color="red")
plt.plot(x_test, y_pred, color="grey")
plt.title("Linear Regresssion")
plt.xlabel("Salary")
plt.ylabel("Year")
plt.show
