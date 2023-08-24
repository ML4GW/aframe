from aframe.utils.timeslides import calc_shifts_required


def test_calc_shifts_required():

    # test that requiring 0 background time returns 0 shifts
    shifts_required = calc_shifts_required(0, 30, 1)
    assert shifts_required == 0

    # need an extra shift to get 60 seconds of background
    # due to the chopping off of livetime at the end of each segment
    shifts_required = calc_shifts_required(60, 30, 1)
    assert shifts_required == 3
