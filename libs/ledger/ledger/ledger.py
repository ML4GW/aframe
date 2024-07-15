import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

import h5py
import numpy as np
from tqdm import tqdm

PATH = Union[str, bytes, os.PathLike]


# define metadata for various types of injection set attributes
# so that they can be easily extended by just annotating your
# new argument with the appropriate type of field
def parameter(default=None):
    default = default or np.array([])
    return field(metadata={"kind": "parameter"}, default=default)


def waveform(default=None):
    default = default or np.array([])
    return field(metadata={"kind": "waveform"}, default=default)


def metadata(default=None):
    return field(metadata={"kind": "metadata"}, default=default)


def _iter_open(files: Iterable[Path], mode: str, clean: bool = True):
    for fname in files:
        with h5py.File(fname, mode) as f:
            yield f
        if clean:
            fname.unlink()


@dataclass
class Ledger:
    def __post_init__(self):
        # get our length up front and make sure that
        # everything that isn't metadata has the same length
        _length = None
        for key, attr in self.__dataclass_fields__.items():
            if attr.metadata["kind"] == "metadata":
                continue
            value = getattr(self, key)

            if _length is None:
                _length = len(value)
            elif len(value) != _length:
                raise ValueError(
                    "Field {} has {} entries, expected {}".format(
                        key, len(value), _length
                    )
                )
        self._length = _length or 0

    def __len__(self):
        return self._length

    def _get_params(self):
        params = []
        for k, v in self.__dataclass_fields__.items():
            if v.metadata["kind"] != "metadata":
                params.append(k)
        return params

    def __iter__(self):
        fields = self._get_params()
        return map(
            lambda i: {k: self.__dict__[k][i] for k in fields},
            range(len(self)),
        )

    # for slicing and masking sets of parameters/waveforms
    def __getitem__(self, *args, **kwargs):
        init_kwargs = {}
        for key, attr in self.__dataclass_fields__.items():
            value = getattr(self, key)
            if attr.metadata["kind"] != "metadata":
                value = value.__getitem__(*args, **kwargs)
                try:
                    len(value)
                except TypeError:
                    value = np.array([value])

            init_kwargs[key] = value
        return type(self)(**init_kwargs)

    def _get_group(self, f: h5py.File, name: str):
        return f.get(name) or f.create_group(name)

    def write(self, fname: PATH, chunks=None) -> None:
        """
        TODO: implement this with an append mode
        """
        with h5py.File(fname, "w") as f:
            f.attrs["length"] = len(self)
            for key, attr in self.__dataclass_fields__.items():
                value = getattr(self, key)

                try:
                    kind = attr.metadata["kind"]
                except KeyError:
                    raise TypeError(
                        f"Couldn't save field {key} with no annotation"
                    )

                if kind == "parameter":
                    params = self._get_group(f, "parameters")
                    params.create_dataset(key, data=value)
                elif kind == "waveform":
                    waveforms = self._get_group(f, "waveforms")
                    waveforms.create_dataset(key, data=value, chunks=chunks)
                elif kind == "metadata":
                    if value is not None:
                        f.attrs[key] = value
                else:
                    raise TypeError(
                        "Couldn't save unknown annotation {} "
                        "for field {}".format(kind, key)
                    )

    @classmethod
    def _load_with_idx(cls, f: h5py.File, idx: Optional[np.ndarray] = None):
        def _try_get(group: str, field: str):
            try:
                group = f[group]
            except KeyError:
                raise ValueError(
                    f"Archive {f.filename} has no group {group}"
                ) from None

            try:
                return group[field]
            except KeyError:
                raise ValueError(
                    "{} group of archive {} has no dataset {}".format(
                        group, f.filename, field
                    )
                ) from None

        kwargs = {}
        for key, attr in cls.__dataclass_fields__.items():
            try:
                kind = attr.metadata["kind"]
            except KeyError:
                raise TypeError(
                    f"Couldn't load field {key} with no 'kind' metadata"
                )

            if kind == "metadata":
                try:
                    value = f.attrs[key]
                except KeyError:
                    value = None
            elif kind not in ("parameter", "waveform"):
                raise TypeError(
                    "Couldn't load unknown annotation {} "
                    "for field {}".format(kind, key)
                )
            else:
                value = _try_get(kind + "s", key)
                if idx is not None:
                    unique_idx, inv_idx = np.unique(idx, return_inverse=True)
                    value = value[unique_idx]
                    value = value[inv_idx]
                else:
                    value = value[:]

            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def read(cls, fname: PATH):
        with h5py.File(fname, "r") as f:
            return cls._load_with_idx(f, None)

    @classmethod
    def sample_from_file(cls, fname: PATH, N: int, replace: bool = False):
        """Helper method for out-of-memory dataloading

        TODO: future extension - add a `weights` callable
        that takes the open h5py.File object as an input
        and computes sampling weights based on parameter values

        Args:
            fname: The file to sample from
            N: The number of samples to draw
            replace:
                Whether to draw with replacement or not.
                If `False`, `N` must be less than the total
                number of samples contained in the file.
        """

        with h5py.File(fname, "r") as f:
            n = f.attrs["length"]
            if N > n and not replace:
                raise ValueError(
                    "Not enough waveforms to sample without replacement"
                )

            # technically faster in the replace=True case to
            # just do a randint but they're both O(10^-5)s
            # so gonna go for the simpler implementation
            idx = np.random.choice(n, size=(N,), replace=replace)
            return cls._load_with_idx(f, idx)

    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        if ours is None:
            return theirs
        elif theirs is None:
            return ours
        elif ours != theirs:
            raise ValueError(
                "Can't append {} with {} value {} "
                "when ours is {}".format(cls.__name__, key, theirs, ours)
            )
        return ours

    def append(self, other) -> None:
        if not isinstance(other, type(self)):
            raise TypeError(
                "unsupported operand type(s) for |: '{}' and '{}'".format(
                    type(self), type(other)
                )
            )

        new_dict = {}
        for key, attr in self.__dataclass_fields__.items():
            ours = getattr(self, key)
            theirs = getattr(other, key)
            if attr.metadata["kind"] == "metadata":
                new = self.compare_metadata(key, ours, theirs)
                new_dict[key] = new
            elif len(ours) == 0:
                new_dict[key] = theirs
            else:
                new_dict[key] = np.concatenate([ours, theirs])

        self.__dict__.update(new_dict)
        self.__post_init__()

    @classmethod
    def aggregate(
        cls,
        files: Iterable[Path],
        fname: Path,
        dtype: np.dtype = np.float64,
        clean: bool = True,
        chunks: Optional[tuple] = None,
    ) -> None:
        """
        Aggregate the data from the files of many smaller
        ledgers into a single larger ledger file
        """
        # iterate through all the files once up front
        # to see how many rows the output ledger will have
        length = 0
        for source in files:
            with h5py.File(source, "r") as f:
                length += f.attrs["length"]

        # now open the output file and start
        # writing to it iteratively
        with h5py.File(fname, "w") as target:
            target.attrs["length"] = length

            idx = 0
            for source in tqdm(_iter_open(files, "r", clean=clean)):
                source_length = source.attrs["length"]
                if source_length == 0:
                    continue
                # for each dataset in the ledger, move the data
                # from the source into the correct spot in the target
                for key, attr in cls.__dataclass_fields__.items():
                    shape = (length,)
                    if attr.metadata["kind"] == "metadata":
                        # for metadata, let compare_metadata decide
                        # how metadata fields should be aggregated
                        # for child classes with special behavior.

                        # if the key is not in the target,
                        # use the default value in the class,
                        # which will be `None` if not specified explicitly
                        if key not in target.attrs:
                            ours = getattr(cls, key)
                        else:
                            ours = target.attrs[key]

                        theirs = source.attrs[key]
                        value = cls.compare_metadata(key, ours, theirs)
                        target.attrs[key] = value
                    else:
                        # get either the parameters or waveforms dataset,
                        # or create it if it doesn't already exist
                        group_name = attr.metadata["kind"] + "s"
                        if group_name not in target:
                            group = target.create_group(group_name)
                        else:
                            group = target[group_name]

                        # grab the source ledger's data, and use its shape
                        # to initialize the dataset in the target if it
                        # does not already exist
                        theirs = source[group_name][key][:]
                        if key not in group:
                            if theirs.ndim > 1:
                                shape += theirs.shape[1:]
                            dataset = group.create_dataset(
                                key, shape=shape, dtype=dtype, chunks=chunks
                            )
                        else:
                            # otherwise grab the target dataset
                            # (but _not_ its presumably large data)
                            dataset = group[key]

                        # now write the source data directly to
                        # the corresponding rows in the target
                        sel = np.s_[idx : idx + source_length]
                        dataset.write_direct(theirs, dest_sel=sel)

                # advance the corresponding row index
                idx += source_length
