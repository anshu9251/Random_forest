import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
data = datasets.load_iris()

#print(data)

#df = pd.DataFrame(data['data'],columns = data['feature_names'])

#df['target'] = data['target']

#print(df.head)
x = data['data']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(X_train.shape)
print(y_train.shape)

print(X_train[:5])
print(X_test[:5])
from sklearn.ensemble import RandomForestClassifier

reg = RandomForestClassifier(n_estimators=100,random_state=42)

reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

new_data = np.array([[5.1, 3.5, 1.4, 0.2],  # Example data point 1
                     [6.2, 2.9, 4.3, 1.3],  # Example data point 2
                     [7.7, 3.0, 6.1, 2.3]]) # Example data point 3

new_prediction  = reg.predict(new_data)

target_name = data['target_names']
predicted_species = [target_name[pred]for pred in new_prediction]

for i,pre_species in enumerate(predicted_species):
    print(f"data point {i+1}:Predicted species :- {pre_species}")
    