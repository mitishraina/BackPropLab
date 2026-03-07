from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love NLP",
    "NLP is fun",
    "I love coding"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())

print("Bag of words matrix:", X.toarray())