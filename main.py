import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('./data.csv')
labels = data.diagnosis
sample = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)

train_sample, test_sample, train_labels, test_labels = train_test_split(sample, labels, test_size=0.3, random_state=42)

cf = RandomForestClassifier(random_state=55, n_estimators=20)
cf = cf.fit(train_sample, train_labels)

model = SelectFromModel(cf, prefit=True)
sample_1 = model.transform(sample)

train_sample_1, test_sample_1, train_labels_1, test_labels_1 = train_test_split(sample_1, labels, test_size=0.3, random_state=42)

cf_1 = RandomForestClassifier(random_state=55, n_estimators=20)
cf_1 = cf_1.fit(train_sample_1, train_labels_1)

ac = accuracy_score(test_labels_1, cf_1.predict(test_sample_1))
print("Accuracy is: ", ac)

cm = confusion_matrix(test_labels_1, cf_1.predict(test_sample_1))
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
