import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import numpy as np
housing = pd.read_csv('housing.csv')
print(housing.ocean_proximity.value_counts())
housing.hist(bins=50,figsize=(10,8))
plt.show()
train_set,test_set = train_test_split(housing,test_size=0.2,random_state = 42)
housing['income_cat'] = pd.cut(housing['median_income'],bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])
housing['income_cat'].hist()
plt.show()
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['income_cat']):
    print(train_index)
    print(test_index)
    strat_train_index = housing.loc[train_index]
    strat_test_index = housing.loc[test_index]
print(strat_test_index['income_cat'].value_counts()/len(strat_test_index))
for set_ in(strat_train_index,strat_test_index):
    set_.drop('income_cat',axis=1,inplace = True)
housing = strat_train_index.copy()
housing.plot(kind="scatter",x="longitude",y='latitude',alpha=0.4,s=housing['population'],
             figsize = (12,8),c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True)
plt.show()
# Finding Corelation
correaltion = housing.corr()
print(correaltion)
print(correaltion.median_house_value.sort_values(ascending=False))

print(housing.columns)
housing = strat_train_index.drop('median_house_value',axis = 1)
housing_label = strat_train_index["median_house_value"].copy()

# filling Missing Values
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median,inplace=True)
housing_num = housing.drop('ocean_proximity',axis=1)
from sklearn.base import  BaseEstimator,TransformerMixin
room_ix,bedroom_ix,population_ix,household_ix = 3,4,5,6
class CombineAttributesAdder( BaseEstimator,TransformerMixin ):
    def __init__(self,add_bedrooms_per_room):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        rooms_per_household = X[:,room_ix]/X[:,household_ix]
        population_per_household = X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedroom_ix]/X[:,household_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('attribes_adder',CombineAttributesAdder()),
    ('std_scaler',StandardScaler())
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.compose import ColumnTransformer
num_attribes = list(housing_num)
cat_attribes = ['ocean_proximity']
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribes),
    ("cat",OneHotEncoder(),cat_attribes),
])
housing_prepared = full_pipeline.fit_transform(housing)
# Apply Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(housing_prepared,housing_label)
data = housing.iloc[:5]
data_prepration = housing_label.iloc[:5]
data_prepration = full_pipeline.transform(data)
print("Prediction",model.predict(data_prepration))