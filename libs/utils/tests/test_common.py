from utils.common import x_per_y


def test_x_per_y():
    # Test when y evenly divides x
    assert x_per_y(9, 3) == 3
    assert x_per_y(20, 5) == 4

    # Test non-even division
    assert x_per_y(10, 3) == 4
    assert x_per_y(21, 5) == 5

    # Test when x is less than y
    assert x_per_y(1, 10) == 1
    assert x_per_y(5, 100) == 1

    # Test when y equals one
    assert x_per_y(1, 1) == 1
    assert x_per_y(100, 1) == 100

    # Test when x equals zero
    assert x_per_y(0, 1) == 0
    assert x_per_y(0, 100) == 0
