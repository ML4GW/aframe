"""
Module defining directory structure conventions for
timeslide analysis. Each `TimeSlide` object represents
a directory containing multiple fields of analysis on
multiple discontiguous segments of data which have all
shifted the strain timeseries of one interferometer by
the same amount. More specifically, `TimeSlide` directory
should be structured like

```
| <root> /
    | field1 /
        | <prefix>-<t0>-<length0>.hdf5
        | <prefix>-<t1>-<length1>.hdf5
        ...
    | field2 /
        | <prefix>-<t0>-<length0>.hdf5
        | <prefix>-<t1>-<length1>.hdf5
        ...
    | field3 /
        | <prefix>-<t0.5>-<length0.5>.hdf5
        ...
```

Where `<root>` should give some indication about the
interferometer shift used to generate the timeslide,
e.g. `"dt-4.5"`. In the future, this nomenclature
may be enforced.

Each field should consist of files containing data
with the same initial timestamp, length, and sample
rate. The length and timestamp of files can vary
within a field, but the sample rate should be
consistent between all of them. If you have some
new timeseries after an analysis with a different
sample rate or initial timestamp from the timeseries
which created it, it should be saved to a new field.

Also consider creating a new field if the new timeseries
represents some analysis which will frequently be
performed asynchronously from the timeseries on which
it depends. This way, if some downstream process needs
this field before it can get started, it's easy to
detect whether it is available or not.

The files within a field are organized into `Segment`
objects, representing ordered files containing fully
contiguous data and which live in the same directory.
All of these constraints are enforced on the creation
of a `Segment` object.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np

from bbhnet.io.h5 import read_timeseries

fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*).hdf5$")
PATH_LIKE = Union[str, Path]
MAYBE_PATHS = Union[PATH_LIKE, Iterable[PATH_LIKE]]


def filter_and_sort_files(
    fnames: MAYBE_PATHS, return_matches: bool = False
) -> List[PATH_LIKE]:
    """Sort data files by their timestamps

    Given a list of filenames or a directory name containing
    data files, sort the files by their timestamps, assuming
    they follow the convention <prefix>-<timestamp>-<length>.hdf5
    """

    if isinstance(fnames, (Path, str)):
        # if we passed a single string or path, assume that
        # this refers to a directory containing files that
        # we're meant to sort
        fname_path = Path(fnames)
        if not fname_path.is_dir():
            raise ValueError(f"'{fnames}' is not a directory")

        fnames = list(fname_path.iterdir())
        fname_it = [f.name for f in fnames]
    else:
        # otherwise make sure the iterable contains either
        # _all_ Paths or _all_ strings. If all paths, normalize
        # them to just include the terminal filename
        if all([isinstance(i, Path) for i in fnames]):
            fname_it = [f.name for f in fnames]
        elif not all([isinstance(i, str) for i in fnames]):
            raise ValueError(
                "'fnames' must either be a path to a directory "
                "or an iterable containing either all strings "
                "or all 'pathlib.Path' objects, instead found "
                + ", ".join([type(i) for i in fnames])
            )
        else:
            fname_it = [Path(f).name for f in fnames]

    # use the timestamps from all valid timestamped filenames
    # to sort the files as the first index in a tuple
    matches = zip(map(fname_re.search, fname_it), fnames)
    tups = [(m.group("t0"), f, m) for m, f in matches if m is not None]

    # if return_matches is True, return the match object,
    # otherwise just return the raw filename
    return_idx = 2 if return_matches else 1
    return [t[return_idx] for t in sorted(tups)]


@dataclass
class Segment:
    fnames: MAYBE_PATHS

    def __post_init__(self):
        if isinstance(self.fnames, (str, Path)):
            self.fnames = [self.fnames]

        # verify that all the indicated filenames exist
        # and come from the same timeslide
        roots = set()
        for f in map(Path, self.fnames):
            if not f.exists():
                raise ValueError(f"Segment file {f} does not exist")
            roots.add(f.parent)

        if len(roots) > 1:
            raise ValueError(
                "Segment files {} come from distinct parents {}".format(
                    ", ".join(map(str, self.fnames)),
                    ", ".join(map(str, roots)),
                )
            )
        root = list(roots)[0]

        # now sort the filenames according to their timestamps
        # and normalize `self.fnames` to be full Path objects
        # TODO: skip if there's only one file?
        matches = filter_and_sort_files(self.fnames, return_matches=True)
        self.fnames = [root / i.string for i in matches]

        # now record some properties of the
        # segment based on the filenames
        # TODO: enforce contiguousness
        self.t0 = float(matches[0].group("t0"))
        self.length = sum([float(i.group("length")) for i in matches])

        # intialize a cache that we can use to load a segment
        # in one process then analyze it in another
        self._cache = {}

    # define some properties and method related to the
    # timespan that a segment encompasses
    @property
    def tf(self):
        return self.t0 + self.length

    def __contains__(self, timestamp):
        return self.t0 <= timestamp < self.tf

    # define properties encoding directory structure conventions
    @property
    def root(self):
        """
        Represents the directory containing all the files from the segment.
        """

        # TODO: can self.fnames be empty?
        if len(self.fnames) == 0:
            return None
        return self.fnames[0].parent

    @property
    def field(self):
        """Represents the field of the timeslide containing this segment"""
        if self.root is None:
            return None
        return self.root.name

    @property
    def shift(self):
        """Represents the TimeSlide root directory"""
        if self.root is None:
            return None
        return self.root.parent.name

    # define methods for manipulating segments
    def append(self, match: Union[PATH_LIKE, re.Match]):
        """Add a new file to the end of this segment

        Accepts `Match` objects from `fname_re` in case
        the `t0` and `length` of this segment have
        already been inferred.
        """

        # if we passed a string or a Path, use fname_re
        # to ensure it's properly formatted and convert
        # it to a match object from which to extract
        # the segments initial timestamp
        if isinstance(match, (str, Path)):
            fname = Path(match)

            # make sure that the file comes from the
            # same directory as the rest of our files
            if fname.parent != self.root:
                raise ValueError(
                    "Can't append filename '{}' to Segment {}".format(
                        fname, self
                    )
                )

            match = fname_re.search(fname.name)
            if match is None:
                raise ValueError(
                    f"Filename '{fname}' not properly formatted "
                    "for addition to timeslide segment."
                )

        # make sure this filename starts off
        # where thesegment currently ends
        fname = self.root / match.string
        if float(match.group("t0")) != self.tf:
            raise ValueError(
                "Filename '{}' has timestamp {} which doesn't "
                "match final timestamp {} of segment {}".format(
                    fname, match.group("t0"), self.tf, self
                )
            )

        # append the filename and increase the length accordingly
        self.fnames.append(fname)
        self.length += float(match.group("length"))

        # reset the cache because we have new data
        self._cache = {}

    def make_shift(self, shift: str) -> "Segment":
        """
        Create a new segment with the same filenames
        from a different timeslide.

        Args:
            shift:
                The root directory of the new timeslide
                to map this Segment's filenames to
        """

        new_fnames = []
        for fname in self.fnames:
            parts = list(fname.parts)
            parts[-3] = shift
            new_fnames.append(Path("/").joinpath(*parts))
        return Segment(new_fnames)

    def read(self, fname, *datasets):
        """
        Thin-as-thread wrapper around read_timeseries to make
        testing the cache simpler via mocking this method.
        """
        return read_timeseries(fname, *datasets)

    def load(self, *datasets) -> Tuple[np.ndarray, ...]:
        """Load the specified fields from this Segment's HDF5 files

        Loads a particular dataset from the files the Segment
        consists of and strings them into a timeseries along
        with the corresponding array of timestamps, returning
        them in the order specified with the timestamps array last.

        Implements a simple caching mechanism that will store
        datasets already requested in a `._cache` dictionary
        which will be consulted before an attempt to load the
        data is made. This makes it easy to analyze a segment
        in multiple processes, while only loading its data
        once up front.
        """

        # first check to see if we have any
        # of the requested datasets cached
        datasets = datasets + ("t",)
        outputs = defaultdict(lambda: np.array([]))
        for dataset in datasets:
            if dataset in self._cache:
                outputs[dataset] = self._cache[dataset]

        # if everything has been cached, then we're done here
        if len(outputs) == len(datasets):
            return tuple(outputs[key] for key in datasets)

        # otherwise load in everything that we didn't have
        fields = [i for i in datasets if i not in outputs]
        read_fields = [i for i in fields if i != "t"]

        for fname in self.fnames:
            # don't specify "t" as a field to read_timeseries
            # because it returns t by default
            values = self.read(fname, *read_fields)

            # append these values to the output field
            # Note that if "t" has been cached, `fields`
            # will be one element shorter than `values`
            # and so the last element, which is `t`,
            # won't get iterated to
            for key, value in zip(fields, values):
                value = np.append(outputs[key], value)
                outputs[key] = value
                self._cache[key] = value

        # return everything in the order requested with time last
        return tuple(outputs[key] for key in datasets)

    def __str__(self):
        return "Segment(root='{}', t0={}, length={})".format(
            self.root, self.t0, self.length
        )


@dataclass
class TimeSlide:
    """
    Object representing the directory structure of a
    particular time-shift of a stretch of (not necessarily
    contiguous) timeseries. Each timeslide is organized into
    mulitple `Segment`s of fully contiguous data which are
    inferred automatically upon initialization.
    """

    root: Union[str, Path]
    field: str

    @property
    def path(self):
        return self.root / self.field

    @property
    def shift(self):
        return self.root.name

    def __post_init__(self):
        self.root = Path(self.root)
        self.update()

    def update(self):
        """Recrawl through the directory and re filter and sort segments"""
        segment = None
        self.segments = []
        for match in filter_and_sort_files(self.path, return_matches=True):
            fname = self.path / match.string

            if segment is None:
                # initialize the first segment if we don't have one yet
                segment = Segment(fname)
            else:
                try:
                    # Otherwise try and append it to the existing segment.
                    # Pass the actual match object so that we don't have
                    # to do an re search to find the t0 of the filename again
                    segment.append(match)
                except ValueError:
                    # if a ValueError got raised, this segment does not
                    # start where the current one ends, so terminate the
                    # current one and start a new one
                    self.segments.append(segment)
                    segment = Segment(fname)

        # append whichever segment was in
        # process when the loop terminated
        self.segments.append(segment)

    @classmethod
    def create(cls, root: Union[Path, str], field: str):
        """Creates a TimeSlide object;
        If path specified by root and field doesn't exist, will
        create it.
        """
        path = root / field
        path.mkdir(exist_ok=True, parents=True)
        obj = cls(root=root, field=field)
        return obj
