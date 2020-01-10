import pandas as pd
import numpy as np
import scipy as sp



# 1. visually inspect data to look for anomalies

df.head()
df.tail()
df.shape
df.info()

    # the 'object' type in pandas is a generic type of data, stored as a string
    # it's lile .file files

# 2. EDA

    # frequency count

df.column_name.value_counts(dropna = False)
    # dropna=False: counts NaNs and report it
    # OR:
df['column_name'].value_counts(dropna = False)

    # summary statistics by columns of numerical types
df.describe()

    # scatter plots: good for visualizing two columns with numbers
df.plot(kind='scatter')


# 3. Data Tidying

    # fix columns containing values instead of variables
pd.melt(frame = df, id_vars = 'id_of_variables', value_vars= ['level_1', 'level_2', 'level_n'],
        var_name = 'variable_name', value_name = 'value_name')

    # pivoting data using df.pivot()
    # when there are duplicate entries on the index and the column at the same time, the data cannot be pivoted, so use
        # pivot_table() instead: pivot_table(..., aggfunc = np.mean, ...) to take mean of duplicate values
df_tidy = df.pivot(values = 'value', index = 'column_to_be_rows', columns = 'row__to_be_columns')


# create columns by slicing entries in columns that contain compound values

# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')
# Create the 'str_split' column
ebola_melt['str_split'] = ebola_melt['type_country'].str.split('_')
# Create the 'type' column
ebola_melt['type'] = ebola_melt.get('str_split').str[0]
# Create the 'country' column
ebola_melt['country'] = ebola_melt.get('str_split').str[1]
# Print the head of ebola_melt
print(ebola_melt.head())

# 4. Combining DataFrames

    # concatenating data
concatenated_df = pd.concat([list of dfs to concat], ignore_index=True)
    # ignore_index=True here will re-index the concatenated df rows

    # globbing: pattern matching for file names

import glob
list_data_dep = []
csv_files = glob.glob('*.csv')
for filename in csv_files:
    data = pd.read_csv(filename)
    list_data_dep.append(data)
concatenated_df = pd.concat(list_data_dep, ignore_index=True)
concatenated_df.to_csv(path_or_buf='/Users/mark/Desktop/panel_demo.csv', index=False)
    # merging: like joining tables in SQL
pd.merge(left = left_table, right = right_table, on = 'columns with the same name, or None if the names are different',
         left_on = 'left table column name', right_on = 'right table column name')


# 5. Converting Data Types

df['column name'] = df['column name'].astype(target_type)
# e.g.
df['column name'] = df['column name'].astype('category')
df['column name'] = df['column name'].astype(str)

    # why convert categorical data into type 'category' objects:
    # 1. make the df smaller in memory
    # 2. make utilizable by other python libraries for analyses

    # dtype str indicates a column of bad data that needs cleaning usually

df['numeric column name'] = pd.to_numeric(df['numeric column name'], errors = 'coerce')
    # errors = 'coerce' will make missing values NaN


# 6. Regular Expression

import re
pattern = re.compile('pattern')
result = re.match(pattern, string)
bool(result)

    # re.findall() returns a list

# 7. apply() in Python

df.apply()

# 8. Drop Duplicate Data

df.drop_duplicates()

# 9. Fill Missing Values

    # user-defined value:
    df['column'] = df['column'].fillna('value of choice')
    # fill multiple columns with 0:
    df[['column 1', 'column2']] = df[['column 1', 'column2']].fillna(0)

    # if there are outliers, median is the best statistic to use

    # check for stuff using assert statement: returns nothing if true, returns error if false
    assert 1 == 1
    assert (df['column'] >= 1).all()
    assert (df >= 1).all().all()

df = pd.DataFrame([[2, 3, 4],[4, 5, 6]])
pd.notnull(df).all().all()