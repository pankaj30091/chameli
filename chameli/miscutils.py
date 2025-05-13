import numpy as np


class DotDict(dict):
    """Dot notation access to dictionary attributes."""

    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return DotDict(value)
        elif isinstance(value, list):
            return [DotDict(item) if isinstance(item, dict) else item for item in value]
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]


def convert_to_dot_dict(dictionary):
    dot_dict = DotDict()
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dot_dict[key] = convert_to_dot_dict(value)
        elif isinstance(value, list):
            dot_dict[key] = [
                convert_to_dot_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            dot_dict[key] = value
    return dot_dict


def np_ffill(arr: np.array, axis: int = 0) -> np.array:
    """Forward fill values in numpy array

    Args:
        arr (np.array): input array with nans
        axis (int, optional): Defaults to 0.

    Returns:
        np.array: numpy array with forward filled values
    """
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [
        np.arange(k)[
            tuple(
                [
                    slice(None) if dim == i else np.newaxis
                    for dim in range(len(arr.shape))
                ]
            )
        ]
        for i, k in enumerate(arr.shape)
    ]
    slc[axis] = idx
    return arr[tuple(slc)]
