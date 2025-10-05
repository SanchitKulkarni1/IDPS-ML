import pickle

# Load the training features used for the multiclass model
with open("models/X_train_multi_selected.pkl", "rb") as f:
    X_train_multi = pickle.load(f)

print("Number of features:", len(X_train_multi.columns))
print("Feature names:\n", list(X_train_multi.columns))
