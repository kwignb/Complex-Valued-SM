from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
from cvmsm import ComplexValuedMSM
from prepro import load_dataset, load_dataset_ex, monotone, hilbert_transform, monotone_i

X_train, X_test, y_train, y_test = load_dataset_ex("cifar10")

# X_train, X_test = monotone(X_train, X_test)

X_train, X_test = monotone_i(X_train, X_test)

# X_train, X_test = hilbert_transform(X_train, X_test)

np.random.seed(3141592653)

param_list = [i for i in range(1,6)]

best_score = 0
best_parameter = {}

for n_subdims in tqdm(param_list):
    n_model = ComplexValuedMSM(n_subdims=n_subdims)
    scores = cross_val_score(n_model, X_train, y_train, cv=5)

    # use average of k evaluate value
    score = np.mean(scores)

    if score > best_score:
        best_score = score
        best_parameter = {'n_subdims': n_subdims}

n_model = ComplexValuedMSM(**best_parameter)
n_model.fit(X_train, y_train)
pred = n_model.predict(X_test)
test_score = accuracy_score(pred, y_test)

print('Best score on validation set: {}'.format(best_score))
print('Best parameter: {}'.format(best_parameter))
print('Test set score with best parameters: {}'.format(test_score))