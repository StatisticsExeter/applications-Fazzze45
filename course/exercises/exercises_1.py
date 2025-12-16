def sum_list(numbers):
    """Return the sum of all elements in the list 'numbers'."""
    return sum(numbers)


def first_of_tuple(t):
    """Return the first element of the tuple 't'."""
    return t[0]


def has_key(d, key):
    """Return True if 'key' exists in dictionary 'd', else False."""
    return key in d


def round_float(f):
    """Round the float 'f' to 2 decimal places."""
    return round(f, 2)


def reverse_list(lst):
    """Return a new list that is the reverse of 'lst'."""
    return lst[::-1]


def count_occurrences(lst, item):
    """For a list of items 'lst', count how many times element 'item' occurs."""
    return lst.count(item)


def tuples_to_dict(pairs):
    """Convert a list of (key, value) tuples 'pairs' into a dictionary."""
    return dict(pairs)


def string_length(s):
    """Return the number of characters in string 's'."""
    return len(s)


def unique_elements(lst):
    """Return a list of unique elements from 'lst'."""
    return list(set(lst))


def swap_dict(d):
    """Return a new dictionary with keys and values of 'd' swapped."""
    return {v: k for k, v in d.items()}
