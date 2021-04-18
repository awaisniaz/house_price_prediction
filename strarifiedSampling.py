import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  StratifiedShuffleSplit
housing = pd.read_csv('housing.csv')
train_Set,test_Set = train_test_split(housing,test_size=0.20,random_state=42)
print(train_Set.shape)
print(test_Set.shape)
print(housing.shape)

housing['income_cat'] = pd.cut(housing['median_income'],bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])
housing['income_cat'].hist()
plt.show()

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_x,test_x in split.split(housing,housing['income_cat']):
    print(train_x)
    print(test_x)
    strat_train_set = housing.loc[train_x]
    strat_test_set = housing.loc[test_x]

for set in (strat_train_set,strat_test_set):
    set.drop('income_cat',axis=1,inplace=True)


housing_data = strat_train_set.copy()
print(housing_data.head())
print(housing_data.columns)

housing_data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing_data['population']/100, label='population',
figsize=(12, 8), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()


