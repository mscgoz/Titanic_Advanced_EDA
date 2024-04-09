import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

"""
    1. Step: Big Picture
"""


def check_df(dataframe, head = 5):
    print("########################################################### Shape ################################################################")
    print(dataframe.shape)
    print("########################################################### Types ################################################################")
    print(dataframe.dtypes)
    print("########################################################### Head ################################################################")
    print(dataframe.head(head))
    print("########################################################### Tail ################################################################")
    print(dataframe.tail(head))
    print("########################################################### NA ################################################################")
    print(dataframe.isnull().sum())
    print("########################################################### Quantiles ################################################################")
    print(dataframe.describe([0, 0.05, 0.50, 0.75, 1]).T)

check_df(df)

"""
    2. Step: Variable Analysis
"""


def grab_col_names(dataframe, cat_th = 10, car_th = 30):
    cat_col = [col for col in df.columns if str(df[col].dtypes) in ["object", "bool", "category"]]
    num_but_cat = [col for col in df.columns if str(df[col].dtypes) in ["int64", "float64"] and df[col].nunique() < 10]
    cat_but_car = [col for col in df.columns if str(df[col].dtypes) in ["object", "category"] and df[col].nunique() > 20]
    cat_col = cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]
    num_col = [col for col in df.columns if str(df[col].dtypes) in ["int64", "float64"] and col not in cat_col]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_col)}")
    print(f"num_cols: {len(num_col)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_col, num_col, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
    3. Step: Categorical and Numerical Variables Summary
"""
# categorical variables summary

def cat_summary(dataframe, col_name):
    print(pd.DataFrame(
        {
            col_name :  dataframe[col_name].value_counts(),
            "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)
            }))
    print("################################")


for col in cat_cols:
    cat_summary(df, col)

# numerical variables summary

def num_summary(dataframe, numerical_columns):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    return print(dataframe[numerical_columns].describe(quantiles).T)


for col in num_cols:
    num_summary(df, col)


"""
    4. Step: Analysis of Target Variable w/ Categorical Variables
"""


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN:": dataframe.groupby(categorical_col)[target].mean()}), end = "\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "survived", col)



"""
    4. Step: Analysis of Target Variable w/ Numerical Variables
"""

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end = "\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "survived", col)
