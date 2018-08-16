# lab6_iris_logisticregression
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

iris = datasets.load_iris()
data = iris.data
target = iris.target
logisticRegression = LogisticRegression()

# cross_val_score (each one against others)
# cv : cross validation
score = model_selection.cross_val_score(logisticRegression, data, target, cv=5)
print("using logistic for iris, score=%s"%(score))
