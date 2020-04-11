from flask import Flask, render_template, request
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import pickle


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        data = []
        name = request.form['name']
        review = request.form['review']
        review = re.sub('[a-zA-Z]/', " ", review)
        review = re.sub('\W', " ", review)
        review = word_tokenize(review)
        review = [WordNetLemmatizer().lemmatize(i) for i in review if i not in set(stopwords.words('english'))]
        review = " ".join(review)
        review = sent_tokenize(review)
        cv = CountVectorizer()
        review = cv.fit_transform(review).toarray()
        review = review.flatten()
        if len(review) < 7427:
            review = np.pad(review, (0, 7427-len(review)), 'constant')
        scaler = pickle.load(open(r'C:\Users\JIMMY\PycharmProjects\oneplus_sentimental_analysis\seriel_data\scaler.pkl', 'rb'))
        review = scaler.transform(review.reshape(1, -1))
        pca = pickle.load(open(r'C:\Users\JIMMY\PycharmProjects\oneplus_sentimental_analysis\seriel_data\pca.pkl', 'rb'))
        review = pca.transform(review)
        model = pickle.load(open(r'C:\Users\JIMMY\PycharmProjects\oneplus_sentimental_analysis\seriel_data\model.pkl', 'rb'))
        prediction = model.predict(review)
        # data.append(name)
        # data.append(prediction)
        data = {'name':name, 'prediction':prediction}

        return render_template('page1.html', data=data)
    else:
        return render_template('page1.html')

if __name__ == '__main__':
    app.run(debug=True)


