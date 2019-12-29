"""
    date:       2019/12/27 7:29 下午
    written by: neonleexiang
"""


# import list:
import os
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


HOUSING_PATH = "datasets/housing"       # path of storage


# function to change the tar file into csv file
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# function to fill the numerical data
def numerical_imputer(data, _strategy='median'):
    imputer=Imputer(strategy=_strategy)
    imputer.fit(data)
    X = imputer.transform(data)
    return pd.DataFram(X, columns=data.columns)


# or we can use the pipeline function to do the jobs of imputer and standardscaler
# housing = load_housing_data()
# housing_num = housing.drop('ocean_proximity', axis=1)
# num_attribs = list(housing_num)
# cat_attribs = ['ocean_proximity']

simple_num_pipeline = Pipeline([
    ('imputer', Imputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])


# ----------------- create attributes -----------------------
from sklearn.preprocessing import FunctionTransformer

housing = load_housing_data()

rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]


def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

# ------------------ DataFrameSelector -----------------------
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    # def print(self):
    #     print(self.attribute_names)

    def transform(self, X):
        #print(X[self.attribute_names].values)
        return X[self.attribute_names].values
# ------------------------------------------------------------


# then we create a full pipeline
from sklearn.compose import ColumnTransformer

def full_pipeline_for_housing_data(data):
    cat_attribs = ["ocean_proximity"]
    num_attribs = list(data.drop('ocean_proximity', axis=1))

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy='median')),
        # ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
        # ('label_binarizer', LabelBinarizer()),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])

    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data


# ------------------------- we should create a whole method for data preprocesing -------------
def data_preprocessing(data, random_state=42):
    """

    :param data: input data
    :return:    type: nparray
            training_data_prepared: trainning data without labels
            training_data_labels: the labels of training data
            test_data_prepared:
            test_data_labels:
    """
    data['income_cat'] = np.ceil(data['median_income'] / 1.5)
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    for train_index, test_index in split.split(data, data['income_cat']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    for set in (strat_train_set, strat_test_set):
        set.drop(['income_cat'], axis=1, inplace=True)

    training_data_prepared = strat_train_set.drop("median_house_value", axis=1)
    training_data_prepared = full_pipeline_for_housing_data(training_data_prepared)
    training_data_labels = strat_train_set['median_house_value'].copy()

    test_data_prepared = strat_test_set.drop("median_house_value", axis=1)
    test_data_prepared = full_pipeline_for_housing_data(test_data_prepared)
    test_data_labels = strat_test_set['median_house_value'].copy()

    # return nparray
    return training_data_prepared, training_data_labels, test_data_prepared, test_data_labels


if __name__ == '__main__':
    housing = load_housing_data()

    # # ----------------- create extra attributes --------------------------
    # housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
    # housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
    # housing['population_per_household'] = housing['population'] / housing['households']
    #
    # # check the correlation
    # print(housing.corr()['median_house_value'].sort_values(ascending=False))

    # ---------------- check the numerical data --------------------------
    # # use imputer to fill the Nah data
    # housing_num = housing.drop('ocean_proximity', axis=1)   # create an copy without attribute 'ocean_proximity'
    # # housing_num_tr = numerical_imputer(housing_num) # numerical transformed data
    # housing_num_tr = simple_num_pipeline.fit_transform(housing_num)
    # print(housin_num_tr.head())

    # ---------------- use full pipeline method ----------------------------
    # first we should divide the test_label
    # housing_labels = housing['median_house_value'].copy()
    # housing = housing.drop('median_house_value', axis=1)
    #
    # housing_prepared_nparray = full_pipeline_for_housing_data(housing)
    #
    # housing_prepared = pd.DataFrame(housing_prepared_nparray)
    # print(housing_prepared.head())
    # print(housing_prepared.info())

    # function = DataFrameSelector(list(housing_num))
    # function.print()
    # function.fit(housing)
    # function.print()
    # function.transform(housing)
    print(data_preprocessing(housing))
