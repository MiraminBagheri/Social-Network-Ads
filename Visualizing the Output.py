# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Outputs.csv')
print(dataset)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2:5].values

# Visualising the Outputs (results)
sns.lineplot(X[:,0], y[:,0], hue=X[:,1])
for i in range(len(X[:,0])):
    if X[:,1][i] == 'Yes' :
        plt.scatter(X[:,0][i],y[:,0][i], color='blue', s=100)
for i in range(len(X[:,0])):
    if X[:,1][i] == 'No' :
        plt.scatter(X[:,0][i],y[:,0][i], color='red')
plt.title('Accuracy of Different modules outpouts')
plt.xlabel('Modules')
plt.ylabel('Accuracy')
plt.xticks(rotation=60)
plt.legend(title="Feature Scaling")
plt.show()

# Visualising the Outputs (results)
sns.lineplot(X[:,0], y[:,1], hue=X[:,1])
for i in range(len(X[:,0])):
    if X[:,1][i] == 'Yes' :
        plt.scatter(X[:,0][i],y[:,1][i], color='blue', s=100)
for i in range(len(X[:,0])):
    if X[:,1][i] == 'No' :
        plt.scatter(X[:,0][i],y[:,1][i], color='red')
plt.title('Standard Deviation of Different modules outpouts')
plt.xlabel('Modules')
plt.ylabel('Standard Deviation')
plt.xticks(rotation=60)
plt.legend(title="Feature Scaling")
plt.show()

# Visualising the Outputs (results)
sns.lineplot(X[:,0], y[:,2], hue=X[:,1])
for i in range(len(X[:,0])):
    if X[:,1][i] == 'Yes' :
        plt.scatter(X[:,0][i],y[:,2][i], color='blue', s=100)
for i in range(len(X[:,0])):
    if X[:,1][i] == 'No' :
        plt.scatter(X[:,0][i],y[:,2][i], color='red')
plt.title('Test-Set Accuracy Score of Different modules outpouts')
plt.xlabel('Modules')
plt.ylabel('Test-Set Accuracy Score')
plt.xticks(rotation=60)
plt.legend(title="Feature Scaling")
plt.show()

# Print Performances and Summary 
print('\n Performances:\n',dataset)
summary1 = {'':['Minimum','Maximum'],
              'Accuracy':[y[:-2,0].min(), y[:-2,0].max()],
              'Module_Name':[X[:-2,0][list(y[:-2,0]).index(y[:-2,0].min())], X[:-2,0][list(y[:-2,0]).index(y[:-2,0].max())]],
              'Feature_Scaling':[X[:-2,1][list(y[:-2,0]).index(y[:-2,0].min())], X[:-2,1][list(y[:-2,0]).index(y[:-2,0].max())]]}
summary1 = pd.DataFrame(summary1, index=None)
print('\n Summary:\n',summary1)
summary2 = {'':['Minimum','Maximum'],
             'Standard Deviation':[y[:-2,1].min(), y[:-2,1].max()],
             'Module_Name':[X[:-2,0][list(y[:-2,1]).index(y[:-2,1].min())], X[:-2,0][list(y[:-2,1]).index(y[:-2,1].max())]],
             'Feature_Scaling':[X[:-2,1][list(y[:-2,1]).index(y[:-2,1].min())], X[:-2,1][list(y[:-2,1]).index(y[:-2,1].max())]]}
summary2 = pd.DataFrame(summary2, index=None)
print('\n Summary:\n',summary2)
summary3 = {'':['Minimum','Maximum'],
             'Test-Set Accuracy Score':[y[:,2].min(), y[:,2].max()],
             'Module_Name':[X[:,0][list(y[:,2]).index(y[:,2].min())], X[:,0][list(y[:,2]).index(y[:,2].max())]],
             'Feature_Scaling':[X[:,1][list(y[:,2]).index(y[:,2].min())], X[:,1][list(y[:,2]).index(y[:,2].max())]]}
summary3 = pd.DataFrame(summary3, index=None)
print('\n Summary:\n',summary3)
