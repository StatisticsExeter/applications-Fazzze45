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

def count_occurrences(lst, value):
    """Return how many times `value` appears in the list `lst`."""
    return lst.count(value)

def list_to_string(lst):
    """Convert a list of items into a single space-separated string."""
    return " ".join(str(item) for item in lst)