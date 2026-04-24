from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer
