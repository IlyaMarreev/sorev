import pandas as pd
import numpy
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
f = open("sample_submission.csv", "w")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#factors that will predict the price
desired_factors = ['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

#set my model to DecisionTree
model = DecisionTreeRegressor()

#set prediction data to factors that will predict, and set target to SalePrice
train_data = train[desired_factors]
test_data = test[desired_factors]
target = train.Target

#fitting model with prediction data and telling it my target
model.fit(train_data, target)

model.predict(test_data.head())
print(model.predict(test_data.head()))
print(model.predict(test_data))
array = model.predict(test_data)
count = 1
result = 'Index' + ', ' + 'Target' 
f.write(result + '\n')
for x in array:
    result = '{}{}{}'.format(str(count),',',str(int(x)))
    f.write(result + '\n')
    count += 1
