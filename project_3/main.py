import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Accident_test.csv")

#NULL CHECK
for i in range(17):
    print("The number of null values in column")
    nullCheck = df.iloc[ : , i].isnull().sum()
    print(nullCheck)

df = df.dropna()

#NULL CHECK again
for i in range(17):
    print("The number of null values in column")
    nullCheck = df.iloc[ : , i].isnull().sum()
    print(nullCheck)

df.Hour_of_Collision.unique()
df.Junction_Control.unique()

df = df.reset_index(drop = True)
print(df)

dummies_1 = pd.get_dummies(df['Weekday_of_Collision'])
print(dummies_1)

dummies_2 = pd.get_dummies(df['Policing_Area'])
print(dummies_2)


df = df.drop(['Policing_Area', 'Weekday_of_Collision'], axis = 'columns', inplace = True)
df = pd.concat([df, dummies_1, dummies_2], axis = 'columns')

print(df)

df_Y = df['Collision_Severity']

df_X = df.drop(['Collision_Severity', 'Collision_Ref_No'], axis = 'columns')


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, train_size = 0.8, random_state = 10)


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_predicted = dt.predict(X_test)

print("The first 5 predicted values are: ")
print(Y_predicted[0:5])
print("The first 5 test values are: ")
print(np.array(Y_test[0:5]))

print(dt.score(X_test, Y_test))

from sklearn import metrics
matrix = metrics.confusion_matrix(Y_predicted, Y_test)
sns.heatmap(matrix, cmap = 'Oranges', annot = True, fmt = '0.1f')
plt.xlabel("Predicted Values")
plt.ylabel("Test values")
plt.title("Confusion Matrix for DecisionTreeClassifier")
plt.show()

from sklearn.model_selection import cross_val_score

dt.fit(df_X, df_Y)
print(cross_val_score(dt, df_X, df_Y, cv = 10))
print(np.average(cross_val_score(dt, df_X, df_Y, cv = 10)))