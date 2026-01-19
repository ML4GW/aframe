import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

PATH = str | bytes | os.PathLike


# define metadata for various types of injection set attributes
# so that they can be easily extended by just annotating your
# new argument with the appropriate type of field
def parameter(default=None):
    """Create a parameter field with metadata for Ledger dataclasses.

    Args:
        default: Optional factory function returning default value.
            Defaults to empty numpy array.

    Returns:
        A dataclass field with parameter metadata.
    """
    default = default or (lambda: np.array([]))
    return field(metadata={"kind": "parameter"}, default_factory=default)


def waveform(default=None):
    """Create a waveform field with metadata for Ledger dataclasses.

    Args:
        default: Optional factory function returning default value.
            Defaults to empty numpy array.

    Returns:
        A dataclass field with waveform metadata.
    """
    default = default or (lambda: np.array([]))
    return field(metadata={"kind": "waveform"}, default_factory=default)


def metadata(default=None, default_factory=None):
    """Create a metadata field with metadata for Ledger dataclasses.

    Args:
        default: Default value for the field. Can only specify one of
                 default or default_factory.
        default_factory: Callable to generate default value.

    Returns:
        A dataclass field with metadata kind annotation.
    """
    # can only specify one of default or default_factory
    kwargs = {}
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    else:
        kwargs["default"] = default
    return field(metadata={"kind": "metadata"}, **kwargs)


def _iter_open(files: Iterable[Path], mode: str, clean: bool = True):
    """Iterate over HDF5 files, opening and optionally removing them.

    Args:
        files: Iterable of file paths to open.
        mode: File open mode ('r', 'w', 'a', etc.).
        clean: If True, delete files after yielding. Defaults to True.

    Yields:
        Opened h5py.File objects.
    """
    for fname in files:
        with h5py.File(fname, mode) as f:
            yield f
        if clean:
            fname.unlink()


@dataclass
class Ledger:
    """Base class for managing collections of parameters and waveforms.

    Provides functionality for data storage, retrieval, sorting,
    and file I/O operations for structured collections of data.
    """

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
        return (
            {k: self.__dict__[k][i] for k in fields} for i in range(len(self))
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

    def is_sorted_by(self, attr: str):
        value = getattr(self, attr)
        if len(value) != len(self) or len(value.shape) > 1:
            raise ValueError(
                "Sorting key should be a 1D array with length equal "
                f"to the length of the object. {attr} has shape "
                f"{value.shape}, and so does not meet these conditions"
            )
        return (value[:-1] <= value[1:]).all()

    def sort_by(self, attr: str):
        if self.is_sorted_by(attr):
            warnings.warn(
                f"Already sorted by {attr}, object is unchanged", stacklevel=2
            )
        idx = np.argsort(getattr(self, attr))
        return self[idx]

    def write(
        self, fname: PATH, chunks: tuple[int, ...] | None = None
    ) -> None:
        """Write ledger data to HDF5 file.

        Args:
            fname: Path to output HDF5 file.
            chunks: Optional tuple specifying chunk shape for
                waveform datasets.

        Raises:
            TypeError: If a field is missing metadata annotation.
        """
        with h5py.File(fname, "w") as f:
            f.attrs["length"] = len(self)
            for key, attr in self.__dataclass_fields__.items():
                value = getattr(self, key)

                try:
                    kind = attr.metadata["kind"]
                except KeyError as exc:
                    raise TypeError(
                        f"Couldn't save field {key} with no annotation"
                    ) from exc

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
    def _load_with_idx(cls, f: h5py.File, idx: np.ndarray | None = None):
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
            except KeyError as exc:
                raise TypeError(
                    f"Couldn't load field {key} with no 'kind' metadata"
                ) from exc

            if kind == "metadata":
                try:
                    value = f.attrs[key]
                except KeyError:
                    value = None
            elif kind not in ("parameter", "waveform"):
                raise TypeError(
                    "Couldn't load unknown annotation {} for field {}".format(
                        kind, key
                    )
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
        """Read ledger data from HDF5 file.

        Args:
            fname: Path to HDF5 file to read.

        Returns:
            Instance of the ledger class populated with data from file.
        """
        with h5py.File(fname, "r") as f:
            return cls._load_with_idx(f, None)

    @classmethod
    def sample_from_file(cls, fname: PATH, N: int, replace: bool = False):
        """Sample data from HDF5 file for out-of-memory operations.

        Loads a random sample of records from a ledger file without
        loading the entire file into memory.

        Args:
            fname: Path to HDF5 file to sample from.
            N: Number of samples to draw.
            replace: If True, draw with replacement. If False, N must be
                less than total samples in file. Defaults to False.

        Returns:
            Instance of the ledger class with sampled data.

        Raises:
            ValueError: If replace=False and N exceeds file length.
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
                "Can't append {} with {} value {} when ours is {}".format(
                    cls.__name__, key, theirs, ours
                )
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
        chunks: tuple[int, ...] | None = None,
        length: int | None = None,
    ) -> None:
        """Aggregate multiple ledger files into a single file.

        Combines data from multiple smaller ledger files into a single
        larger file, with optional cleanup of source files.

        Args:
            files: Iterable of file paths to aggregate. All files must be
                   structured as expected by the `cls` object.
            fname: Output file path for the aggregated ledger.
            dtype: Datatype for storing parameters and waveforms data.
                   Defaults to np.float64.
            clean: If True, remove source files after aggregation.
                   Defaults to True.
            chunks: Optional tuple specifying chunk shape for efficient
                    waveform reading.
            length: Total length of final Ledger object. If unspecified,
                    determined from source file metadata. Providing this
                    in advance saves a read operation per source file.
        """
        if length is None:
            # iterate through all the files once up front
            # to see how many rows the output ledger will have
            length = 0
            for source in tqdm(files):
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
                            # try to access the metadata default value
                            try:
                                ours = getattr(cls, key)
                            except AttributeError:
                                # if not specified, use the default factory
                                ours = attr.default_factory()
                        else:
                            ours = target.attrs[key]

                        theirs = source.attrs[key]
                        value = cls.compare_metadata(key, ours, theirs)
                        target.attrs[key] = value
                    else:
                        # only chunk waveforms, not parameters
                        _chunks = (
                            chunks
                            if attr.metadata["kind"] == "waveform"
                            else None
                        )
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
                                key, shape=shape, dtype=dtype, chunks=_chunks
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
