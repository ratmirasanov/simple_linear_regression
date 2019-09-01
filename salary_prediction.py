import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


DATA_SET = pd.read_csv("salary.csv")
X = DATA_SET.iloc[:, :-1].values
Y = DATA_SET.iloc[:, 1].values

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=1/3, random_state=0)

REGRESS_OR = LinearRegression()
REGRESS_OR.fit(X_TRAIN, Y_TRAIN)

Y_PREDICTION = REGRESS_OR.predict(X_TEST)

plt.scatter(X_TRAIN, Y_TRAIN, color="blue")
plt.plot(X_TRAIN, REGRESS_OR.predict(X_TRAIN), color="red")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_TEST, Y_TEST, color="blue")
plt.plot(X_TRAIN, REGRESS_OR.predict(X_TRAIN), color="red")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
