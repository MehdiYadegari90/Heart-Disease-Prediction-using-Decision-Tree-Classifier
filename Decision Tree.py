


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load the dataset
df = pd.read_csv('heart.csv')

# Show the distribution of the target variable (output)
# Uncomment to see the counts of each class in the 'output' column
# print(df["output"].value_counts())

# Optional: Visualize the distribution of cholesterol levels
# df.hist(column="chol", bins=50)
# plt.show()

# Define feature variables (X) and target variable (y)
# Features used for prediction
x = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
         'exng', 'oldpeak', 'slp', 'caa', 'thall']].values
# Target variable
y = df['output'].values

# Encoding categorical variables (if necessary)
# Uncomment and modify the following lines if you need to encode labels
# le_sex = preprocessing.LabelEncoder()
# le_sex.fit(["M", "F"])
# x[:, 1] = le_sex.transform(x[:, 1])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Print shapes of the train and test sets for verification
# print("Train set=", x_train.shape, y_train.shape)
# print("Test set=", x_test.shape, y_test.shape)

# Create and train the Decision Tree classifier
heartTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
heartTree.fit(x_train, y_train)

# Make predictions on the test set
y_hat = heartTree.predict(x_test)

# Print the accuracy of the model on the test set
print("Test set Accuracy=", metrics.accuracy_score(y_test, y_hat))
