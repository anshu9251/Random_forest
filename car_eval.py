import pandas as pd

col_names = ["Buying_price","maint_cost","Doors","Persons","luggage_size","safety","class"]
file_path = "D:\car_evaluation.csv"
df = pd.read_csv(file_path,header=None,names=col_names)


#print(df)

#print(df[:5])
#print(df.info())

feature_names = list(df.columns.values[:-1])

for feature in feature_names:
    unq_count = df[feature].nunique()
    unq_val = df[feature].unique()

    #print("{}: {} values, {}".format(feature,unq_count,unq_val))

label_names = df.columns.values[-1]

#print("{}: {} values, {}".format(label_names,df[label_names].nunique(),df[label_names].unique())) 

#print(df[label_names].value_counts())

df_encoded = pd.get_dummies(df,columns=feature_names,drop_first=True,dtype=int)

#print(df_encoded.tail())

df_encoded['class'], class_uniques = pd.factorize(df_encoded['class'])


class_col = df_encoded.pop("class")

df_encoded["class"] = class_col
print(df_encoded.tail())


#Random forest classifier

X = df_encoded.loc[:,"Buying_price_low" : "safety_med"]
y = df_encoded.loc[:,"class"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,max_depth=None,max_features="sqrt",min_samples_leaf=1,min_samples_split=2,random_state=42)

rf.fit(X_train,y_train)

#test our model

y_pred = rf.predict(X_test)

print(y_pred[0:5])

y_pred_labelled = class_uniques[y_pred]
print(y_pred_labelled[:5])

print(rf.predict_proba(X_test)[:5])

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(accuracy_score(y_test,y_pred))

y_pred = class_uniques[y_pred]
y_test = class_uniques[y_test]

print(confusion_matrix(y_test,y_pred ,labels = ["unacc","acc","good","vgood"]))

print(classification_report(y_test,y_pred,labels=["unacc","acc","good","vgood"]))


#ordinal data

df_ordinal = df.copy()

print(df_ordinal.head())

df_ordinal["buying_ordinal"]= df_ordinal["Buying_price"].map({"low":0,"med":1,"high":2,"vhigh":3})
df_ordinal["maint_ordinal"] = df_ordinal["maint_cost"].map({"low":0,"med":1,"high":2,"vhigh":3})
df_ordinal["doors_ordinal"]= df_ordinal["Doors"].map({"2":0,"3":1,"4":2,"5more":3})
df_ordinal["persons_ordinal"]=df_ordinal["Persons"].map({'2':0, '4':1, 'more':2})
df_ordinal["luggage_ordinal"]=df_ordinal["luggage_size"].map({"small":0,"med":1,"big":2})
df_ordinal["safety_ordinal"]=df_ordinal["safety"].map({'low':0, 'med':1, 'high':2})
df_ordinal["class_ordinal"]=df_ordinal["class"].map({'unacc':0, 'acc':1, 'good':2, 'vgood':3})

df_ordinal.drop(columns=["Buying_price","maint_cost","Doors","Persons","luggage_size","safety","class"],inplace=True)

print(df_ordinal.tail())

X_ord = df_ordinal.iloc[:,:-1]
y_ord = df_ordinal.iloc[:,-1]

X_ord_train,X_ord_test,y_ord_train,y_ord_test = train_test_split(X_ord,y_ord,test_size=0.25,random_state=42)

rf_ord = RandomForestClassifier(n_estimators=100,max_depth=None,max_features="sqrt",min_samples_leaf=1,min_samples_split=2,random_state=42)

rf_ord.fit(X_ord_train,y_ord_train)
    
y_ord_pred = rf_ord.predict(X_ord_test)

print(accuracy_score(y_ord_test,y_ord_pred))
print(confusion_matrix(y_ord_test,y_ord_pred,labels=[0,1,2,3]))
print(classification_report(y_ord_test,y_ord_pred,labels=[0,1,2,3]))

"""from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators":[100,150,200],
    "max_depth":[None,10,20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2',None]
}

gd = GridSearchCV(RandomForestClassifier(random_state=42),param_grid=param_grid,cv=5,scoring="accuracy")

gd.fit(X_ord_train,y_ord_train)

best_rf = gd.best_estimator_
test_accuracy = best_rf.score(X_ord_test,y_ord_test)
best_params = gd.best_params_

#print(test_accuracy)
print(best_params)  """
