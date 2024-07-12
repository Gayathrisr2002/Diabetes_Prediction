#import library
import pandas as pd
# Read dataset
df=pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Diabetes.csv')
# df.columns
# define y and X
y=df['diabetes']
X=df[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'dpf', 'age']]
#split to train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2529)
#Model selection
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
#training
model.fit(X_train,y_train)
#Prediction
y_pred=model.predict(X_test)

#Evaluation
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
import pickle
name='model.sav'
pickle.dump(model,open(name,'wb'))

load_model=pickle.load(open('model.sav','rb'))