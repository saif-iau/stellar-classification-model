

# Pandas is used for data manipulation
import pandas as pd# Read in data and display first 5 rows
df = pd.read_csv('star_classification.csv')
df.head(100)
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing, tree, datasets
from dtreeviz.trees import dtreeviz

pd.set_option('display.max_rows', 10)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)

classifier = tree.DecisionTreeClassifier(class_weight=None, 
                                         criterion='entropy', 
                                         max_depth=3,
                                         max_features='sqrt', 
                                         splitter='best', 
                                         random_state=24)
model = classifier.fit(X_train, y_train)
model.score(X_test, y_test)


features_dict = {'feature_importances': classifier.feature_importances_, 'feature_names': X_train.columns}

pd.DataFrame(features_dict).sort_values(by='feature_importances', ascending=False).head(19)

df2 = df
df2.drop(columns=['fiber_ID','delta','u','r','z','obj_ID','run_ID','rerun_ID','cam_col','field_ID','spec_obj_ID','MJD','g','i','plate','alpha'],inplace=True)
df2.head(2)

X2 = df2.drop('class', axis=1)
y2 = df2['class']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=24)

classifier2 = tree.DecisionTreeClassifier(class_weight=None, 
                                         criterion='entropy', 
                                         max_depth=3,
                                         max_features='sqrt', 
                                         splitter='best', 
                                         random_state=24)
model2 = classifier2.fit(X_train2, y_train2)
model2.score(X_test2, y_test2)
@app.route('/')
def home():
        return render_template('index.html')



@app.route('/predict', methods=['POST','GET'])
def predict():
    redshift = request.form.get('redshift')
    y_predict = classifier2.predict([[redshift]])[0]
    return render_template('index.html',result = y_predict)



if __name__ == '__main__':
    app.run(port=3000,debug=True)  
  
