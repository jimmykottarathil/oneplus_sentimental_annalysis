from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import pickle

review = 'fffffffff'
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
scaler = pickle.load(open(r'seriel_data/scaler.pkl', 'rb'))
review = scaler.transform(review.reshape(1,-1))
pca = pickle.load(open(r'seriel_data/pca.pkl', 'rb'))
review = pca.transform(review)
model = pickle.load(open(r'seriel_data/model.pkl', 'rb'))
prediction = model.predict(review)
print(prediction)