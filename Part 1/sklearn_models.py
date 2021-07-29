
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the train_data.csv file
# The file contains the training data
# The data is in the form of a csv file
# The first column is the label
# The rest of the columns are features
# The features are in the form of numbers

test = pd.read_csv("./data/test_data.csv")
data = pd.read_csv("./data/train_data.csv")
labels = pd.read_csv("./data/train_labels.csv")
train_data_df = data.drop(['id'], axis=1)
train_labels = labels.drop(['id'], axis=1)

# Split the data into training and validation data (80%-20%)
train_data, val_data, train_labels, val_labels= train_test_split(train_data_df, train_labels, test_size=0.2)

model = RandomForestClassifier()
model.fit(train_data, train_labels.values.ravel())
print('Train accuracy of Random Forest Classifier:',accuracy_score(model.predict(train_data), train_labels))
print('Val accuracy of Random Forest Classifier:', accuracy_score(model.predict(val_data), val_labels))

knn = KNeighborsClassifier()
knn.fit(train_data, train_labels.values.ravel())
print('Train accuracy score of K Neighbours:',accuracy_score(knn.predict(train_data), train_labels))
print('Val accuracy score of K Neighbours:', accuracy_score(knn.predict(val_data), val_labels))

test_data = test.drop(['id'], axis=1)
randomforestpred = model.predict(test_data)
knnpred = knn.predict(test_data)

# add column to test data
test['Random Forest Pred'] = randomforestpred
test['KNN Pred'] = knnpred
test_pred = test[['id','Random Forest Pred', 'KNN Pred']]

# save the predict result to csv file
test_pred.to_csv('./data/sklearn_models_pred.csv', index=False)