from sklearn.decomposition import PCA

def reduce_features(X):
    pca = PCA(n_components=150)  # adjust if needed
    X_pca = pca.fit_transform(X)
    return X_pca, pca