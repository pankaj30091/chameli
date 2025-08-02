import numpy as np

# Import chameli_logger lazily to avoid circular import
def get_chameli_logger():
    """Get chameli_logger instance to avoid circular imports."""
    from . import chameli_logger
    return chameli_logger


class DotDict(dict):
    """Dot notation access to dictionary attributes with enhanced logging."""

    def __getattr__(self, item):
        try:
            value = self.get(item)
            if isinstance(value, dict):
                return DotDict(value)
            elif isinstance(value, list):
                return [DotDict(item) if isinstance(item, dict) else item for item in value]
            return value
        except Exception as e:
            get_chameli_logger().log_error("Failed to get attribute from DotDict", e, {
                "item": item,
                "available_keys": list(self.keys())
            })
            raise

    def __setattr__(self, key, value):
        try:
            self[key] = value
            get_chameli_logger().log_debug("Set attribute in DotDict", {
                "key": key,
                "value_type": type(value).__name__
            })
        except Exception as e:
            get_chameli_logger().log_error("Failed to set attribute in DotDict", e, {
                "key": key,
                "value_type": type(value).__name__
            })
            raise

    def __delattr__(self, item):
        try:
            del self[item]
            get_chameli_logger().log_debug("Deleted attribute from DotDict", {
                "item": item
            })
        except Exception as e:
            get_chameli_logger().log_error("Failed to delete attribute from DotDict", e, {
                "item": item,
                "available_keys": list(self.keys())
            })
            raise


def convert_to_dot_dict(dictionary):
    """Convert a dictionary to a DotDict with enhanced logging."""
    try:
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
            
            get_chameli_logger().log_debug("Successfully converted dictionary to DotDict", {
                "input_keys": list(dictionary.keys()) if isinstance(dictionary, dict) else None,
                "output_keys": list(dot_dict.keys())
            })
        return dot_dict
        
    except Exception as e:
        get_chameli_logger().log_error("Failed to convert dictionary to DotDict", e, {
            "input_type": type(dictionary).__name__,
            "input_keys": list(dictionary.keys()) if isinstance(dictionary, dict) else None
        })
        raise


def np_ffill(arr: np.array, axis: int = 0) -> np.array:
    """Forward fill values in numpy array with enhanced logging.

    Args:
        arr (np.array): input array with nans
        axis (int, optional): Defaults to 0.

    Returns:
    Returns:
        np.array: numpy array with forward filled values
    """
    # Validate input
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if axis < 0 or axis >= len(arr.shape):
        raise ValueError(f"Axis {axis} is out of bounds for array with {len(arr.shape)} dimensions")
        
    # Log operation details
    get_chameli_logger().log_debug("Starting forward fill operation", {
        "axis": axis,
        "nan_count": np.isnan(arr).sum(),
        "total_elements": arr.size
    })
    try:        
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
        result = arr[tuple(slc)]
            
        # Log success
        get_chameli_logger().log_debug("Forward fill operation completed successfully", {
            "result_shape": result.shape,
            "remaining_nans": np.isnan(result).sum(),
            "filled_count": np.isnan(arr).sum() - np.isnan(result).sum()
        })
        
        return result
        
    except Exception as e:
        get_chameli_logger().log_error("Failed to perform forward fill operation", e, {
            "array_shape": arr.shape if hasattr(arr, 'shape') else None,
            "axis": axis,
            "array_type": type(arr).__name__
        })
        raise
