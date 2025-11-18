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