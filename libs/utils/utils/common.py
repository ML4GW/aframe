def x_per_y(x: int, y: int) -> int:
    """
    Calculate the ceiling division of x by y.

    Computes how many groups of size y are needed to contain x items,
    which is equivalent to ceil(x / y). This is useful for determining
    the number of batches or chunks needed to process a dataset.

    Args:
        x (int): The numerator (total number of items).
        y (int): The denominator (group size).

    Returns:
        int: The ceiling of x divided by y.

    Examples:
        >>> x_per_y(10, 3)
        4
        >>> x_per_y(9, 3)
        3
        >>> x_per_y(1, 5)
        1
    """
    # Using integer arithmetic to avoid floating point precision issues
    return int((x - 1) // y) + 1
