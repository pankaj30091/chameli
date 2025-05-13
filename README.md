# chameli

chameli is a lightweight Python library providing utilities for working with holidays, dates, options, and other trading-related tools. It is designed to simplify common tasks in quantitative finance and trading workflows.

---

## Features

- **Date Utilities**: Simplify date and time operations.
- **Options Pricing**: Tools for European options calculations.
- **External Interactions**: Functions for assisting web scraping, reading and writing R files.
- **Config Management**: Centralized configuration management using YAML files.

---

## Installation

Install the package using pip:

```bash
pip install chameli
```

---

## Functions Overview

Below is a list of available functions in each module. For detailed usage, use the `help(function_name)` command in Python.

### `dateutils` Module
- `load_holidays_by_exchange()`
- `valid_datetime(sdatetime, out_pattern=None)`
- `is_business_day(date, exchange="NSE")`
- `business_days_between(start_date, end_date, include_first=False, include_last=False, exchange="NSE")`
- `calc_fractional_business_days(start_datetime, end_datetime, exchange="NSE")`
- `advance_by_biz_days(datetime_, days, adjustment="fbd", exchange="NSE")`
- `get_last_day_of_month(year, month)`
- `get_expiry(date, weekly=0, day_of_week=4, exchange="NSE")`
- `is_aware(datetime_)`
- `get_aware_dt(datetime_, tz="Asia/Kolkata")`
- `get_naive_dt(datetime_)`
- `is_time_between(begin_time, end_time, check_time)`

### `europeanoptions` Module
- `BlackScholesPrice(S, X, r, sigma, T, OptionType)`
- `BlackScholesDelta(S, X, r, sigma, T, OptionType)`
- `BlackScholesGamma(S, X, r, sigma, T)`
- `BlackScholesVega(S, X, r, sigma, T)`
- `BlackScholesTheta(S, X, r, sigma, T, OptionType, daysInYear=252)`
- `BlackScholesIV(S, X, r, T, OptionType, OptionPrice, seed=0.2)`
- `calc_delta(long_symbol, opt_price, underlying, time, days_in_year=252, risk_free_rate=0, exchange="NSE")`
- `get_option_price(long_symbol, S, sigma, calc_time, r=0, days_in_year=252, exchange="NSE")`
- `calc_greeks(long_symbol, opt_price, underlying, calc_time, greeks, days_in_year=252, risk_free_rate=0, exchange="NSE")`
- `generate_opt_simulation(symbols, quantities, target_date, vol_shift, start_range, end_range, increment, market_data, exchange="NSE")`
- `find_x_intercepts(price, value)`

### `interactions` Module
- `readRDS(filename)`
- `saveRDS(pd_file, path)`
- `send_mail(send_from, send_to, password, subject, text, files=None, is_html=False)`
- `get_session_or_driver(url_to_test, get_session=True, headless=False, desktop_session=4, proxy_source=None, api_key=None, proxy_user=None, proxy_password=None, country_code=None, webdriver_path=None)`

### `miscutils` Module
- `convert_to_dot_dict(dictionary)`
- `np_ffill(arr, axis=0)`

---
