"""
    date:       2019/12/27 7:23 下午
    written by: neonleexiang
"""


# import list:
import os
import pandas as pd     # use pandas to extract the data into dataframe
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


HOUSING_PATH = "datasets/housing"       # path of storage


# function to change the tar file into csv file
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    housing = load_housing_data()

    print(housing.head())   # review of the dataframe
    print(housing.info())   # check the value of dataframe
    print(housing["ocean_proximity"].value_counts())    # print the info of 'ocean_proximity' value

    # ------------------------ matplotlib usage -------- data visualization ---------------------------------
    housing.hist(bins=50, figsize=(20, 15))     # hist plot of every attributes
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)    # plot scatter
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                 s=housing['population'] / 100, label="population",
                 c=housing['median_house_value'], cmap=plt.get_cmap('jet'), colorbar=True,
                 )      # plot the scatter to show the relationship between price as well as population and location
    plt.legend()    # plot the legend

    plt.show()
    # ------------------------- end of data visualization ---------------------------------------------------


    # ------------------------------ find the relationship between each attributes ----------------------------
    corr_matrix = housing.corr()    # correlation matrix
    print(corr_matrix)

    # also we can plot it by using pandas tools
    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(housing[attributes], figsize=(12, 8))

    housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
    # ------------------------------------------------------------------------------------------------------


