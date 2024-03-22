import h5py
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries


def find_max_times(path: str, n: int = 1) -> np.ndarray:
    """
    path, path to background.h5 file
    n, n highest maxima (e.g. n=2, top 2 highest detection statistic)

    return central times times for n highest detection statistics
    given background path
    """

    with h5py.File(path, "r") as f:
        params = f["parameters"]
        ds = params["detection_statistic"][:]
        central_time = params["time"][:]
        shift = params["shift"][:]

    ind = np.argsort(ds)[-n:]
    ind = np.sort(ind)
    top_times = central_time[
        ind
    ]  # central times corresponding to highest detection statistic(s)
    top_shift = shift[ind]  # shift L1 data by this amount

    if n > len(ds):
        raise ValueError(
            "{} > length of detection array. Choose a smaller integer".format(
                n
            )
        )
    else:
        return top_times, top_shift


def load_background(times: np.ndarray, shifts: np.ndarray, window: float):
    """
    Load in background.hdf5 data from data/test/background dir
    shifts: list of shifts for L1 data
    window: duration of data to transform over
    """

    for counter, t in enumerate(times):
        t = int(t)
        shift = int(shifts[counter][1])
        H1 = TimeSeries.fetch_open_data("H1", start=t - window, end=t + window)
        H1 = H1.resample(2048)
        L1 = TimeSeries.fetch_open_data(
            "L1", start=t + shift - window, end=t + shift + window
        )
        L1 = L1.resample(2048)

        qgrams_H1 = H1.q_transform()
        plot_H1 = qgrams_H1.plot()
        plt.gca().set_title("H1 Omegascan")
        plot_H1.savefig(
            "omega_scan_H1_{time}_{number}.png".format(time=t, number=counter)
        )

        qgrams_L1 = L1.q_transform()
        plot_L1 = qgrams_L1.plot()
        plt.gca().set_title("L1 Omegascan")
        plot_L1.savefig(
            "omega_scan_L1_{time}_{number}.png".format(time=t, number=counter)
        )


def main(n: int, background_file: str, window: float):
    times, shifts = find_max_times(background_file, n)
    load_background(times=times, shifts=shifts, window=window)


if __name__ == "__main__":
    background_file = (
        "/home/ricco.venterea/results/my-first-run/infer/background.h5"
    )
    n = 2
    main(n, background_file, 2)
