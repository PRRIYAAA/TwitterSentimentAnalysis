from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Load CSV
df = pd.read_csv('twitter_dataset.csv') 

# Map numerical labels to string labels
label_mapping = {0: 'negative', 4: 'positive'}
df['sentiment'] = df['sentiment'].map(label_mapping)

# Convert text and labels to list
texts = df['text'].astype(str).tolist()
labels = df['sentiment'].astype(str).tolist()

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    vectorized = vectorizer.transform([input_text])
    prediction = model.predict(vectorized)[0]

    emoji = {
        'positive': '😊✨',
        'negative': '😞💔',
        'neutral': '😐'
    }.get(prediction, '')

    prediction_with_emoji = f"{prediction.capitalize()} {emoji}"
    return render_template('index.html', prediction=prediction_with_emoji, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
