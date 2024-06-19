## Spam yorumları tespit etme

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_csv("../data/youtube-spam.csv")
# print(data.sample(5))

data = data[["CONTENT", "CLASS"]]
# Class = 0 spam değil, class = 1 Spam
# print(data.sample(5))

data["CLASS"] = data["CLASS"].map({0: "Spam Değil",
                                   1: "Spam Yorum"})
print(data.sample(5))

"""
Bu problem ikili sınıflandırma problemi olduğundan, modeli eğitmek için Bernoulli Naive Bayes algoritmasını kullanacağım.

"""

x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

model = BernoulliNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# sample = "Check this out: https://thecleverprogrammer.com/" 
# data = cv.transform([sample]).toarray()
# print(model.predict(data))

# sample = "Lack of information!" 
# data = cv.transform([sample]).toarray()
# print(model.predict(data)) 

sample = "!!dsadasdas" 
data = cv.transform([sample]).toarray()
print(model.predict(data)) 

