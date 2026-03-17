import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scr.logistic_regression import LogisticRegressionScratch
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler


#dataset load
df = pd.read_csv("data/logisiticRegression.csv")
df.drop(columns=["User ID"],axis=1,inplace=True)
# feature and target
df["Gender"] = df["Gender"].map({"Male":0,"Female":1})
x = df[["Gender","Age","EstimatedSalary"]].values
y = df["Purchased"].values

plt.boxplot(df["EstimatedSalary"])
plt.title("Box Plot of Estimated Salary")
plt.ylabel("Salary")
plt.show()
x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=42,test_size=0.2)

lr =StandardScaler()
x_train_scale=lr.fit_transform(x_train)
x_test_scale=lr.transform(x_test)
print(x_test_scale,x_train_scale)
model =LogisticRegressionScratch(lr=0.01,epochs=1000)
model.fit(x_train_scale,y_train)
y_pred =model.predict(x_test_scale)
print("Accuracy:", accuracy_score(y_test, y_pred))

# first 10 predictions
print("First 10 Predictions:", y_pred[:10])
print("First 10 Actual:", y_test[:10])

# model parameters
print("Weights:", model.weights)
print("Bias:", model.bias)
