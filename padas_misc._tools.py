import pandas as pd

# df accessor
# .loc uses labels and .iloc uses indices
    # i.e. .loc uses words while .iloc uses numbers

# pd.read_csv(index_col = 'blah') helps set a column in the file as the index column

# to access columns, use df['column name']
# to access rows, use df.loc['row name or index']

# df.loc['A':'B', 'C':'D'] where A:B are rows and C:D are columns
# lists can also be used in place of slices, to select specific and non-consecutive rows or columns

# use double square brackets [[]] to return a dataframe, and single square brackets to return a series

# reverse slicing: .loc['b':'a':-1], where a is upstairs b

# use a boolean series to select a subset of a dataframe, i.e. filtering

# df.dropna(thresh=, ) thresh: the threshold of np.nans, drop along the axis the entries with more than a number
    #of np.nans specified by the threshold

# df.apply()
# df_celsius = weather.loc[:,['Mean TemperatureF', 'Mean Dew PointF']].apply(to_celsius)

# .map() transforms values according to a dictionary
# election['color'] = election.winner.map(red_vs_blue)
# the index can be more than one column

# df.set_index(['column 1', 'column 2'])
# df.sort_index()

# to access all of one level of a Multiindex, use slice(None)
# all_month2 = sales.loc[(slice(None),2),:]

# df.unstack(level = 'row index level to unstack') will turn the row index level to columns

# df.groupby(['list', 'of row indices'])

# df.agg(func=)

# always use .sort_index() after passing multiple columns into index_col= 
