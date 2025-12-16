import pandas as pd


def sum_list(numbers):
    """Given a list of integers 'numbers' return the sum of this list."""
    return sum(numbers)


def max_value(numbers):
    """Given a list of numbers 'numbers' return the maximum value of this list."""
    return max(numbers)


def reverse_string(string):
    """Given a string 'string' return the reversed version of the input string."""
    return string[::-1]


def filter_even(numbers):
    """Given a list of numbers 'numbers' return a list containing only the even numbers from the input list."""
    return [n for n in numbers if n % 2 == 0]


def get_fifth_row(df):
    if len(df) < 5:
        raise IndexError("Not enough rows")
    return df.iloc[4]


def column_mean(df, column):
    """Given a dataframe 'df' and the name of a column 'column' return the mean of the specified column."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame")

    if df[column].isna().all():
        return float('nan')

    return df[column].mean()


def lookup_key(d, key):
    """Given a dictionary 'd' and a key 'key' return the value associated with the key in the dictionary."""
    return d.get(key)


def count_occurrences(lst):
    """Given a list 'lst' return a dictionary with counts of each unique element in the list."""
    return {x: lst.count(x) for x in set(lst)}


def drop_missing(df):
    """Given a dataframe 'df' with some rows containing missing values, return a dataframe with rows containing missing values removed."""
    return df.dropna()


def value_counts_df(df, column):
    """Given a dataframe 'df' with various columns and the name of one of those columns 'column', return a dataframe with value counts of the specified column."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    return counts
