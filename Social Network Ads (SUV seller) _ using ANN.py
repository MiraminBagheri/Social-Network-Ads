# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)
# print(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=512, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=256, activation='relu'))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=256, activation='relu'))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=256, activation='relu'))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=256, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

# Training the ANN on the Training set
ann.load_weights('Weights.h1')
# ann.fit(X_train, y_train, batch_size = 4, epochs = 200)
# ann.save_weights('Weights.h1')

# Part 4 - Making the predictions and evaluating the model

# Predicting a new result
print(ann.predict(sc.transform([[30,87000]])) > 0.5)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('\n Test-Set Accuracy Score: ', accuracy_score(y_test, y_pred))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.015625),
                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 32))
plt.contourf(X1, X2, ann.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color = ListedColormap(('red', 'green'))(i), label = j)
plt.title('ANN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.015625),
                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 32))
plt.contourf(X1, X2, ann.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color = ListedColormap(('red', 'green'))(i), label = j)
plt.scatter(30, 87000, color = 'yellow' if ann.predict(sc.transform([[30,87000]])) == 0 else 'blue', marker='*', label='Single Pred', s=100)
plt.title('ANN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Save the output (result) in CSV file :
Header = ['Module_Name', 'Feature Scaled', 'Accuracy', 'Standard Deviation', 'Test-Set Accuracy Score']
output= [['ANN', 'Yes', '', '', accuracy_score(y_test, y_pred)]]
output = np.array(output)
Output = pd.DataFrame(output)
try :
    pd.read_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Social Network Ads (SUV seller)/Outputs.csv')
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Social Network Ads (SUV seller)/Outputs.csv', mode='a', index=False, header=False)
except:
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Social Network Ads (SUV seller)/Outputs.csv', mode='w', sep=',', index=False, header=Header)
