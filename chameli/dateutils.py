import calendar
import datetime as dt
import logging
from typing import Union

import numpy as np
import pandas as pd
import pytz

from .config import get_config


# Import chameli_logger lazily to avoid circular import
def get_chameli_logger():
    """Get chameli_logger instance to avoid circular imports."""
    from . import chameli_logger

    return chameli_logger


pd.options.display.float_format = "{:.2f}".format


def get_dynamic_config():
    return get_config()


def get_timezone_by_exchange(exchange: str):
    """Get timezone for the given exchange from the dynamic config."""
    return get_dynamic_config().get("markets", {}).get(exchange, {}).get("timezone", "Asia/Kolkata")


def load_holidays_by_exchange():
    """Load holidays for each exchange from the config.yaml file and convert them to dt.date."""

    holidays_by_exchange = {}
    for exchange, details in get_dynamic_config().get("markets", {}).items():
        holiday_dates = details.get("holidays", [])
        # Convert holiday strings to dt.date objects
        holidays_by_exchange[exchange] = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in holiday_dates]
    return holidays_by_exchange


def load_timings_by_exchange():
    """Load market open and close timings for each exchange from the config.yaml file.

    This function retrieves the market open and close times for each exchange
    defined in the `markets` section of the configuration file. The times are
    converted to `datetime.time` objects for easier manipulation.

    Returns:
        dict: A dictionary where the keys are exchange names and the values are
              dictionaries containing `open_time` and `close_time` as `datetime.time` objects.
    """
    timings_by_exchange = {}
    for exchange, details in get_dynamic_config().get("markets", {}).items():
        open_time_str = details.get("open_time")
        close_time_str = details.get("close_time")
        tz = details.get("timezone")

        # Convert open and close times to `datetime.time` objects
        if open_time_str and close_time_str and tz:
            try:
                timings_by_exchange[exchange] = {
                    "open_time": open_time_str,
                    "close_time": close_time_str,
                    "timezone": tz,
                }
            except ValueError as e:
                get_chameli_logger().log_error(
                    f"Invalid time format for exchange {exchange}", e, {"exchange": exchange}
                )
        else:
            get_chameli_logger().log_warning(
                f"Missing open or close time for exchange {exchange}", {"exchange": exchange}
            )

    return timings_by_exchange


holidays = load_holidays_by_exchange()
market_timings = load_timings_by_exchange()


def valid_datetime(sdatetime, out_pattern=None):
    """Parses and validates a given datetime string or object against multiple patterns.

    This function attempts to parse a datetime string or object (`sdatetime`) using a predefined
    set of patterns. If the input is already a `datetime` or `date` object, it can optionally
    format it into a string using the specified `out_pattern`. If parsing or formatting fails,
    the function returns `(False, False)`.

        sdatetime (str | datetime.datetime | datetime.date):
            The input datetime string or object to validate and parse.
        out_pattern (str, optional):
            The desired output format for the datetime object. Defaults to None.

        tuple:
            - If successful:
                - A `datetime` object or formatted string (if `out_pattern` is provided).
                - The pattern used for parsing (or `None` if `sdatetime` is already a datetime object).
            - If unsuccessful:
                - `(False, False)` indicating the input could not be parsed or formatted.
    """
    for pattern in [
        "%d-%b-%Y",
        "%d-%b-%Y %H:%M:%S",
        "%Y%m%d%H%M%S",
        "%d-%b-%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
        "%d-%m-%Y %H:%M:%S%z",
        "%Y%m%d",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y%m%d %H:%M:%S",
        "%Y%m%d-%H%M%S",
    ]:
        try:
            if isinstance(sdatetime, (dt.datetime, dt.date)):
                if not out_pattern:
                    if isinstance(sdatetime, pd.Timestamp):
                        return sdatetime.to_pydatetime(), None
                    else:
                        return sdatetime, None
                else:
                    return dt.datetime.strftime(sdatetime, out_pattern), None

            dt_parsed = dt.datetime.strptime(sdatetime, pattern)
            if out_pattern is not None:
                return dt_parsed.strftime(out_pattern), pattern
            else:
                return dt_parsed, pattern
        except (ValueError, TypeError):
            continue
    return False, False


def is_business_day(date, exchange="NSE"):
    """
    Determine if a given date is a business day for a specified exchange.

    A business day is defined as a weekday (Monday to Friday) that is not a holiday
    for the specified exchange.

    Args:
        date (datetime or str): The date to check. Can be a datetime object or a string
            that can be parsed into a datetime.
        exchange (str, optional): The exchange for which to check the business day.
            Defaults to "NSE" (National Stock Exchange).

    Returns:
        bool: True if the date is a business day for the specified exchange, False otherwise.

    Raises:
        ValueError: If the input date is invalid or cannot be parsed.
    """

    date, _ = valid_datetime(date)
    date = date.date() if isinstance(date, dt.datetime) else date
    return date.weekday() < 5 and date not in holidays.get(exchange)


def business_days_between(start_date, end_date, include_first=False, include_last=False, exchange="NSE"):
    """
    Calculate the number of business days between two dates, optionally including the start and/or end dates.

    Args:
        start_date (str or datetime.date or datetime.datetime): The start date of the range.
            Can be a string in a valid date format or a datetime object.
        end_date (str or datetime.date or datetime.datetime): The end date of the range.
            Can be a string in a valid date format or a datetime object.
        include_first (bool, optional): Whether to include the start date in the count if it is a business day.
            Defaults to False.
        include_last (bool, optional): Whether to include the end date in the count if it is a business day.
            Defaults to False.
        exchange (str, optional): The stock exchange for which business days are calculated.
            Defaults to "NSE".

    Returns:
        int: The number of business days between the start and end dates, adjusted for the include_first
            and include_last options. Returns -1000000 if the input dates are not properly formatted.

    Raises:
        Exception: If the input dates are not in a valid format, an error is logged, and a default error
            value of -1000000 is returned.

    Notes:
        - Business days are defined as weekdays (Monday to Friday) that are not holidays for the specified exchange.
        - The function uses a helper function `generate_business_days` to generate the list of business days
          between the start and end dates.
    """

    def generate_business_days(start_date, end_date, exchange):
        business_days = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() not in [
                5,
                6,
            ] and current_date not in holidays.get(exchange):
                business_days.append(current_date)
            current_date += dt.timedelta(days=1)
        return business_days

    try:
        start_date, _ = valid_datetime(start_date)
        is_datetime = isinstance(start_date, dt.datetime)
        start_date = start_date.date() if is_datetime else start_date
        end_date, _ = valid_datetime(end_date)
        is_datetime = isinstance(end_date, dt.datetime)
        end_date = end_date.date() if is_datetime else end_date
    except Exception as e:
        get_chameli_logger().log_error(
            f"start_date:{start_date}, end_date:{end_date} are not properly formatted!! exiting business day calculation",
            e,
            {"start_date": str(start_date), "end_date": str(end_date)},
        )
        return -1000000
    business_days = generate_business_days(start_date, end_date, exchange)
    num_business_days = len(business_days)
    # Adjust for includeFirst and includeLast options
    if not include_first and start_date in business_days:
        num_business_days -= 1
    if not include_last and end_date in business_days:
        num_business_days -= 1

    return num_business_days


def calc_fractional_business_days(
    start_datetime: Union[str, dt.datetime], end_datetime: Union[str, dt.datetime], exchange="NSE"
) -> float:
    """
    Calculate the fractional number of business days between two datetime strings,
    considering market open and close times.

    This function computes the total number of business days between two timestamps,
    including fractional parts of the first and last days based on the market's
    operating hours. It accounts for weekends and holidays specific to the given exchange.

    Args:
        start_datetime (str): The start datetime in the format "%Y-%m-%d %H:%M:%S".
        end_datetime (str): The end datetime in the format "%Y-%m-%d %H:%M:%S".
        exchange (str, optional): The exchange identifier (e.g., "NSE") to determine
        holidays and market hours. Defaults to "NSE".

    Returns:
        float: The fractional number of business days between the two datetimes.

    Raises:
        ValueError: If the input datetime strings are invalid or if the start time
                    is after the end time.

    Notes:
        - If the time component is missing in the input datetimes, it defaults to
          the market close time.
        - The function uses the `business_days_between` and `is_business_day`
          utilities to account for holidays and weekends.
        - The fractional part of a business day is calculated based on the elapsed
          time relative to the market's operating hours.

    Examples:
        >>> calc_fractional_business_days(
        ...     "2023-03-01 10:00:00",
        ...     "2023-03-03 15:30:00",
        ...     market_open_time="09:00:00",
        ...     market_close_time="15:30:00",
        ...     exchange="NSE"
        ... )
        2.5
    """

    market_open_time = market_timings.get(exchange).get("open_time")
    market_close_time = market_timings.get(exchange).get("close_time")

    try:
        # Validate inputs
        start_date_str, _ = valid_datetime(start_datetime, "%Y-%m-%d %H:%M:%S")
        end_date_str, _ = valid_datetime(end_datetime, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    if not start_date_str or not end_date_str:
        raise ValueError("Both start_time and end_time should be valid timestamps or strings")

    if start_date_str > end_date_str:
        raise ValueError("Start time cannot be after end time")

    # Ensure time component exists, default to market close if missing
    if " " not in start_date_str:
        start_date_str += f" {market_close_time}"
    if " " not in end_date_str:
        end_date_str += f" {market_close_time}"

    # Convert to datetime objects
    start_dt = dt.datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    end_dt = dt.datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
    market_open_dt = dt.datetime.strptime(market_open_time, "%H:%M:%S")
    market_close_dt = dt.datetime.strptime(market_close_time, "%H:%M:%S")

    # Compute whole business days (excluding first)
    biz_days = business_days_between(
        start_dt.date(),
        end_dt.date(),
        include_first=False,
        include_last=True,
        exchange=exchange,
    )
    if biz_days < 0:
        get_chameli_logger().log_error(
            f"End date {end_datetime} cannot be earlier than start date {start_datetime}",
            None,
            {"end_datetime": str(end_datetime), "start_datetime": str(start_datetime)},
        )
        return np.nan

    # Normalize time stubs as fractions of a business day
    def compute_time_stub(time_dt):
        """Returns the fraction of the market day that has passed since market open."""
        if time_dt.time() <= market_open_dt.time():
            return 0  # Before or at market open
        elif time_dt.time() >= market_close_dt.time():
            return 1  # After market close
        else:
            elapsed = (time_dt - dt.datetime.combine(time_dt.date(), market_open_dt.time())).total_seconds()
            total_market_time = (market_close_dt - market_open_dt).total_seconds()
            return elapsed / total_market_time

    # Compute front stub (start day)
    front_stub = 0
    if is_business_day(start_dt, exchange):
        front_stub = 1 - compute_time_stub(start_dt)

    # Compute end stub (end day)
    end_stub = 0
    if is_business_day(end_dt, exchange):
        end_stub = 1 - compute_time_stub(end_dt)

    return biz_days + front_stub - end_stub  # subtract end_stub as biz_days includes end_date


def advance_by_biz_days(
    datetime_: Union[str, pd.Timestamp, dt.datetime, dt.date],
    days: int,
    adjustment: str = "fbd",  # "fbd" for Following Business Day, "pbd" for Preceding Business Day
    exchange: str = "NSE",
) -> Union[str, pd.Timestamp, dt.datetime, dt.date]:
    """
    Advances or adjusts a given date by a specified number of business days,
    considering the business calendar of a specified exchange.

    Args:
        datetime_ (Union[str, pd.Timestamp, dt.datetime, dt.date]):
            The input date to be adjusted. Can be a string, pandas Timestamp,
            Python datetime, or date object.
        days (int):
            The number of business days to advance. Positive values move forward,
            negative values move backward. If `days` is 0, the function adjusts
            the date to the nearest business day based on the `adjustment` parameter.
        adjustment (str, optional):
            Adjustment rule to apply when `days` is 0 and the input date is not
            a business day. Options are:
            - "fbd": Following Business Day (default).
            - "pbd": Preceding Business Day.
        exchange (str, optional):
            The exchange whose business calendar is used to determine business
            days. Default is "NSE".

    Returns:
        Union[str, pd.Timestamp, dt.datetime, dt.date]:
            The adjusted date in the same format as the input `datetime_`.

    Raises:
        ValueError: If the input date format is invalid or unsupported.

    Notes:
        - The function assumes the existence of an `is_business_day` function
          that determines whether a given date is a business day for the specified
          exchange.
        - If the input is a string, it is assumed to follow the format "%Y-%m-%d"
          unless otherwise specified.

    Examples:
        >>> advance_by_biz_days("2023-03-01", 5, exchange="NSE")
        '2023-03-08'

        >>> advance_by_biz_days(pd.Timestamp("2023-03-01"), -3, exchange="NYSE")
        Timestamp('2023-02-24 00:00:00')

        >>> advance_by_biz_days(dt.date(2023, 3, 1), 0, adjustment="pbd", exchange="LSE")
        datetime.date(2023, 2, 28)
    """

    # Parse the input datetime
    # Parse the input date using valid_datetime
    parsed_date, input_format = valid_datetime(datetime_)

    # Determine if the input was a datetime or date
    is_datetime = isinstance(parsed_date, dt.datetime)

    # Normalize to date for processing
    current_date = parsed_date.date() if is_datetime else parsed_date

    # If days > 0 or < 0, move by business days
    if days != 0:
        step = 1 if days > 0 else -1
        business_days_moved = 0

        while business_days_moved != abs(days):
            current_date += dt.timedelta(days=step)
            if is_business_day(current_date, exchange):
                business_days_moved += 1

    elif not is_business_day(current_date, exchange):
        step = 1 if adjustment == "fbd" else -1
        while not is_business_day(current_date):
            current_date += dt.timedelta(days=step)

    # Convert back to the original format
    if isinstance(datetime_, str):
        return current_date.strftime(input_format)
    elif isinstance(datetime_, pd.Timestamp):
        return pd.Timestamp(current_date).tz_localize(parsed_date.tzinfo)
    elif isinstance(datetime_, dt.datetime):
        return dt.datetime.combine(current_date, parsed_date.time()).replace(tzinfo=parsed_date.tzinfo)
    else:
        return current_date


def business_minutes_shift(
    start_time: Union[str, dt.datetime],
    minutes: int,
    exchange: str = "NSE",
) -> dt.datetime:
    """
    Shift a datetime by exactly `minutes` business minutes, skipping non-business periods.
    Positive `minutes` moves forward, negative moves backward.
    """
    # Ensure start_time is a datetime object
    parsed_date, input_format = valid_datetime(start_time)
    is_datetime = isinstance(parsed_date, dt.datetime)
    if not isinstance(parsed_date, dt.datetime):
        raise ValueError("start_time must be a datetime or string convertible to datetime")

    # Get market open/close times from market_timings and convert to dt.time
    market_open_str = market_timings.get(exchange, {}).get("open_time", "09:15:00")
    market_close_str = market_timings.get(exchange, {}).get("close_time", "15:30:00")
    market_open = dt.datetime.strptime(market_open_str, "%H:%M:%S").time()
    market_close = dt.datetime.strptime(market_close_str, "%H:%M:%S").time()

    current = parsed_date
    remaining = abs(minutes)
    direction = 1 if minutes >= 0 else -1

    while remaining > 0:
        # Make open_dt and close_dt timezone-aware if current is aware
        if is_aware(current):
            tzinfo = current.tzinfo
            open_dt = dt.datetime.combine(current.date(), market_open).replace(tzinfo=tzinfo)
            close_dt = dt.datetime.combine(current.date(), market_close).replace(tzinfo=tzinfo)
        else:
            open_dt = dt.datetime.combine(current.date(), market_open)
            close_dt = dt.datetime.combine(current.date(), market_close)

        if direction > 0:
            # Moving forward
            if current < open_dt:
                current = open_dt
            minutes_left_today = int((close_dt - current).total_seconds() // 60)
            if remaining <= minutes_left_today:
                return current + dt.timedelta(minutes=remaining)
            else:
                remaining -= minutes_left_today
                # Move to next business day open
                next_biz_day = advance_by_biz_days(current.date(), 1, exchange)
                current = dt.datetime.combine(next_biz_day, market_open)
        else:
            # Moving backward
            if current > close_dt:
                current = close_dt
            minutes_since_open = int((current - open_dt).total_seconds() // 60)
            if remaining <= minutes_since_open:
                temp = current - dt.timedelta(minutes=remaining)
                if isinstance(start_time, str):
                    return temp.strftime(input_format)
                elif is_datetime:
                    return temp.replace(tzinfo=parsed_date.tzinfo)
                else:
                    return temp
            else:
                remaining -= minutes_since_open
                # Move to previous business day close
                prev_biz_day = advance_by_biz_days(current.date(), -1, exchange)
                current = dt.datetime.combine(prev_biz_day, market_close)
    if isinstance(start_time, str):
        return current.strftime(input_format)
    elif is_datetime:
        return current.replace(tzinfo=parsed_date.tzinfo)
    else:
        return current

    return current


def get_last_day_of_month(year, month):
    """
    Get the last day of a given month and year.

    Args:
        year (int): The year.
        month (int): The month (1-12).

    Returns:
        datetime.date: The last day of the month as a date object.
    """
    # Use calendar.monthrange to get the number of days in the month
    last_day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, last_day)


def apply_timezone(dt_obj, exchange: str):
    """
    Ensure dt_obj is timezone-aware and in the timezone for the given exchange.
    If dt_obj is naive, localize it. If it's aware but in a different timezone, convert it.
    """
    tz = get_timezone_by_exchange(exchange)
    target_tz = pytz.timezone(tz)
    if dt_obj.tzinfo is None:
        return target_tz.localize(dt_obj)
    elif dt_obj.tzinfo != target_tz:
        return dt_obj.astimezone(target_tz)
    return dt_obj


def get_expiry(
    date: Union[str, dt.datetime, dt.date], weekly=0, day_of_week: int = 4, exchange="NSE"
) -> Union[str, dt.datetime, dt.date]:
    """
    Calculate the last working expiry date for a given input date.

    Args:
        date (Union[str, dt.datetime]): Input reference date for which expiry is sought.
        weekly (int, optional): If 0, calculates the last working weekday of the month.
                                If greater than 0, calculates weekly expiries.
        day_of_week (int, optional): Day of the week for expiration. Defaults to 4 (Thursday).
        exchange (str, optional): Exchange holidays to consider. Defaults to NSE.

    Returns:
        Union[str, dt.datetime, dt.date]: Expiry date in the same format as the input.
    """

    def adjust_to_previous_working_day(target_date):
        """
        Adjust the given date to the previous working day if it falls on a holiday or weekend.

        Args:
            target_date (datetime.date): The date to adjust.

        Returns:
            datetime.date: Adjusted working day.
        """
        while target_date in holidays.get(exchange) or target_date.weekday() >= 5:
            target_date -= dt.timedelta(days=1)
        return target_date

    def last_valid_weekday_of_month(year, month, target_weekday):
        """
        Find the last occurrence of the target weekday in the month.
        If the calculated date is a holiday or weekend, move backward to a valid business day.

        Args:
            year (int): Year of the target month.
            month (int): Month of the target weekday.
            target_weekday (int): Desired weekday (1=Monday, 7=Sunday).

        Returns:
            datetime.date: Last valid business day for the specified weekday in the month.
        """
        # Start with the last day of the month
        last_day = get_last_day_of_month(year, month)

        # Traverse backward to find the last occurrence of the specified weekday
        while last_day.weekday() != target_weekday - 1:
            last_day -= dt.timedelta(days=1)

        # Adjust for holidays and weekends
        return adjust_to_previous_working_day(last_day)

    # Parse the input date using valid_datetime
    parsed_date, input_format = valid_datetime(date)

    # Determine if the input was a datetime or date
    is_datetime = isinstance(parsed_date, dt.datetime)

    # Normalize to date for processing
    date_mod = parsed_date.date() if is_datetime else parsed_date

    if weekly > 0:
        # Weekly expiry: Find the Nth weekly expiry from the input date
        def find_next_valid_expiry(start_date):
            """Find the next valid expiry date starting from start_date (inclusive).
            If the target weekday is a holiday/weekend, uses the previous business day."""
            current = start_date
            # Find the next occurrence of the target weekday
            while current.weekday() != day_of_week - 1:
                current += dt.timedelta(days=1)
            # Adjust backward for holidays and weekends - this is the expiry date
            return adjust_to_previous_working_day(current)
        
        # Check if today is a valid expiry day
        is_today_expiry = (date_mod.weekday() == day_of_week - 1 and 
                          date_mod not in holidays.get(exchange))
        
        if is_today_expiry and weekly == 1:
            # Today is an expiry day and we want the first week, so return today
            expiry = date_mod
        else:
            # Find the first expiry
            # If today is an expiry day but weekly > 1, start from tomorrow to skip today
            start_date = date_mod + dt.timedelta(days=1) if is_today_expiry else date_mod
            expiry = find_next_valid_expiry(start_date)
            
            # Calculate how many more expiries to find
            # If today is an expiry day, we've already skipped weekly=1, so find (weekly-2) more
            # If today is not an expiry day, find (weekly-1) more after the first one
            additional_expiries = (weekly - 2) if is_today_expiry else (weekly - 1)
            for _ in range(additional_expiries):
                expiry = find_next_valid_expiry(expiry + dt.timedelta(days=1))

    else:
        # Monthly expiry: Always calculate the last valid weekday of the month
        current_month_expiry = last_valid_weekday_of_month(date_mod.year, date_mod.month, day_of_week)

        # If the input date is beyond the current month's expiry, calculate for the next month
        if date_mod > current_month_expiry:
            next_month = (date_mod.month % 12) + 1
            year = date_mod.year + (next_month == 1)
            current_month_expiry = last_valid_weekday_of_month(year, next_month, day_of_week)

        expiry = current_month_expiry

    # Ensure expiry is >= input date
    while expiry < date_mod:
        expiry += dt.timedelta(days=1)

    # Return the expiry date in the same format as the input
    if isinstance(date, str):
        return expiry.strftime(input_format)
    elif is_datetime:
        combined_date = dt.datetime.combine(expiry, dt.datetime.min.time())
        return combined_date.replace(tzinfo=parsed_date.tzinfo)
    else:
        return expiry


def is_aware(datetime_: Union[pd.Timestamp, dt.datetime]) -> bool:
    """Is the argument aware?

    Args:
        datetime_dt (dt.datetime): datetime.datetime object

    Returns:
        bool: True if aware else False
    """
    if isinstance(datetime_, dt.datetime):
        if datetime_.tzinfo is not None and datetime_.tzinfo.utcoffset(datetime_) is not None:
            return True
        else:
            return False
    elif isinstance(datetime_, pd.Timestamp):
        if datetime_.tz is None:
            return False
        else:
            return True
    else:
        raise ValueError("Input should be either datetime.datetime or pd.Timestamp")


def get_aware_dt(datetime_: Union[dt.datetime, pd.Timestamp], tz="Asia/Kolkata") -> Union[dt.datetime, pd.Timestamp]:
    """
    Converts a naive or timezone-aware datetime object to a timezone-aware datetime object
    in the specified timezone.
    Parameters:
    ----------
    datetime_ : Union[datetime.datetime, pandas.Timestamp]
        The input datetime object, which can be either naive or timezone-aware.
    tz : str, optional
        The target timezone to convert or localize the datetime object to.
        Defaults to "Asia/Kolkata".
    Returns:
    -------
    Union[datetime.datetime, pandas.Timestamp]
        A timezone-aware datetime object in the specified timezone.
    Raises:
    ------
    ValueError
        If the input is not of type `datetime.datetime` or `pandas.Timestamp`.
    Notes:
    ------
    - If the input datetime is naive, it will be localized to the specified timezone.
    - If the input datetime is already timezone-aware, it will be converted to the specified timezone.
    """

    tzLocal = pytz.timezone(tz)
    if isinstance(datetime_, dt.datetime):
        if is_aware(datetime_):
            # convert aware to another timezone
            return datetime_.astimezone(tzLocal)
        else:
            return tzLocal.localize(datetime_)  # localize naive datettime
    elif isinstance(datetime_, pd.Timestamp):
        if is_aware(datetime_):
            # convert aware to tzLocal
            return datetime_.tz_convert(tz)
            return datetime_
        else:
            # convert naive to tzLocal
            return datetime_.tz_localize(tz)
    else:
        raise ValueError("Input should be either datetime.datetime or pd.Timestamp")


def get_naive_dt(datetime_: Union[dt.datetime, pd.Timestamp]) -> Union[dt.datetime, pd.Timestamp]:
    """Remove timezone, if any

    Args:
        datetime_dt (Union[dt.datetime,pd.Timestamp]): datetime, either naive or aware

    Returns:
        Union[dt.datetime,pd.Timestamp]: Remove timezone information, making the datetime naive
    """
    if isinstance(datetime_, dt.datetime):
        if is_aware(datetime_):
            return datetime_.replace(tzinfo=None)
        else:
            return datetime_
    elif isinstance(datetime_, pd.Timestamp):
        if is_aware(datetime_):
            datetime_ = datetime_.tz = None
            return datetime_
        else:
            return datetime_
    else:
        raise ValueError("Input should be either datetime.datetime or pd.Timestamp")


def is_time_between(begin_time: dt.time, end_time: dt.time, check_time: dt.time = dt.datetime.now().time()) -> bool:
    """check if time is in-between. Return false at boundary

    Args:
        begin_time (dt.time): begin time
        end_time (dt.time): end time
        check_time (dt.time, optional): Time to be checked for range condition.If none,
        current system time is considered. Defaults to None.

    Returns:
        bool: True if within strict range.
    """
    # If check time is not given, default to current UTC time
    if begin_time < end_time:
        return check_time > begin_time and check_time < end_time
    else:  # crosses midnight
        return check_time > begin_time or check_time < end_time
