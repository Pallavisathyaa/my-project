from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_section import train_test_split


digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.75)

tpot = TPOTClassifier(verbosity=2)
tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))

