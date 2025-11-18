def sum_list(nums):
    """Return the sum of a list of numbers."""
    return sum(nums)

def multiply_list(nums):
    """Return the product of a list of numbers."""
    result = 1
    for n in nums:
        result *= n
    return result

def reverse_string(s):
    """Return the reverse of a string."""
    return s[::-1]

def max_value(nums):
    """Return the maximum value in a list of numbers."""
    return max(nums)

def filter_even(nums):
    """Return only even numbers from the list."""
    return [n for n in nums if n % 2 == 0]


def get_fifth_row(df):
    """Return the fifth row of a dataframe."""
    return df.iloc[4]


def column_mean(df, column_name):
    """Return the mean of a column in a dataframe."""
    return df[column_name].mean()

def lookup_key(d, key):
    """Return the value for a given key in a dictionary."""
    return d.get(key)

def count_occurrences(lst):
    """Return a dictionary counting occurrences of each element in lst."""
    result = {}
    for item in lst:
        result[item] = result.get(item, 0) + 1
    return result

def list_to_string(lst):
    """Return a comma-separated string from a list of strings."""
    return ",".join(lst)


from datetime import datetime

def parse_date(date_string):
    """Parse a date string in YYYY-MM-DD format and return a datetime.date object."""
    return datetime.strptime(date_string, "%Y-%m-%d").date()