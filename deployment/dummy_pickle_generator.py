# TO BE DELETED AFTER THE REAL PICKLES ARE GENERATED!!!!

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Create dummy training data (mimics real feature dimension)
n_features = 8255  # Typical feature dimension from HOG+Color+LBP
n_samples = 100
X_dummy = np.random.randn(n_samples, n_features)

# Create and FIT the scaler (this is the key step!)
scaler = StandardScaler()
scaler.fit(X_dummy)  # <-- This makes it "fitted"
with open('saved_models/feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create and save dummy SVM
svm = SVC()
svm.fit(X_dummy, np.random.randint(0, 7, n_samples))  # Dummy labels
with open('saved_models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# Create and save dummy KNN
knn = KNeighborsClassifier()
knn.fit(X_dummy, np.random.randint(0, 7, n_samples))  # Dummy labels
with open('saved_models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("Properly fitted dummy files created successfully!")
