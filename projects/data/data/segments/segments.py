from typing import Iterable, Optional

from gwpy.segments import DataQualityDict, DataQualityFlag, SegmentList

OPEN_DATA_FLAGS = ["H1_DATA", "L1_DATA", "V1_DATA"]
O3A_END = 1253977218
O3B_START = 1256655618


class DataQualityDict(DataQualityDict):
    @classmethod
    def query_non_open(
        cls, flags: Iterable[str], start: float, end: float, **kwargs
    ) -> DataQualityDict:
        try:
            return cls.query_dqsegdb(flags, start, end, **kwargs)
        except OSError as e:
            if not str(e).startswith(
                "Could not find the TLS certificate file"
            ):
                # TODO: what's the error for an expired certificate?
                raise

            # try to authenticate then re-query
            # authenticate()
            return cls.query_dqsegdb(flags, start, end, **kwargs)

    @classmethod
    def query_open(
        cls, flags: Iterable[str], start: float, end: float, **kwargs
    ) -> DataQualityDict:
        dqdict = cls()
        for flag in flags:
            dqdict[flag] = DataQualityFlag.fetch_open_data(
                flag, start, end, **kwargs
            )
        return dqdict

    @classmethod
    def _query_segments(
        cls,
        flags: Iterable[str],
        start: float,
        end: float,
        min_duration: Optional[float] = None,
        **kwargs
    ) -> SegmentList:
        flags = set(flags)
        open_flags = set(OPEN_DATA_FLAGS)

        open_data_flags = list(flags & open_flags)
        flags = list(flags - open_flags)

        segments = cls()
        if flags:
            segments.update(cls.query_non_open(flags, start, end, **kwargs))
        if open_data_flags:
            segments.update(
                cls.query_open(open_data_flags, start, end, **kwargs)
            )

        segments = segments.intersection().active
        if min_duration is not None:
            segments = filter(lambda i: i[1] - i[0] >= min_duration, segments)
            segments = SegmentList(segments)
        return segments

    @classmethod
    def query_segments(
        cls,
        flags: Iterable[str],
        start: float,
        end: float,
        min_duration: Optional[float] = None,
        **kwargs
    ) -> SegmentList:
        # if the requested time period
        # spans O3a to O3b, query the two
        # separately and append
        if start < O3A_END and end > O3B_START:
            segments = SegmentList()
            segments.extend(
                cls._query_segments(
                    flags, start, O3A_END, min_duration, **kwargs
                )
            )
            segments.extend(
                cls._query_segments(
                    flags, O3B_START, end, min_duration, **kwargs
                )
            )
            return segments
        # otherwise, just query the whole period
        return cls._query_segments(flags, start, end, min_duration, **kwargs)
