from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load dataset
    data = pd.read_csv('Crop.csv')

    # Encoding labels
    label1 = data.iloc[:, 7]
    label_encoder = LabelEncoder()
    encoded_crops = label_encoder.fit_transform(label1)

    # Training starts
    X = data.iloc[:, 0:7]
    y = encoded_crops

    # Train-test split
    X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.2, random_state=2022)

    # Initialize models
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

     # Train and evaluate each model
    accuracies = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = metrics.accuracy_score(y_val, y_pred)
        accuracies[model_name] = accuracy
        #print(f"Accuracy of {model_name}: {accuracy:.2f}")  # Print the accuracy for each model


    # Predictions with Random Forest (default model for prediction)
    RF = models["Random Forest"]

    # Predictions
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        prediction_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = RF.predict(prediction_data)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        return render_template('result.html', predicted_crop=predicted_crop)

if __name__ == '__main__':
    app.run(debug=True)
