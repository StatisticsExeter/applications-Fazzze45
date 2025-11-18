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