import datetime as dt
import logging
import math
import sys
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .config import get_config
from .dateutils import calc_fractional_business_days, valid_datetime, apply_timezone
from .miscutils import convert_to_dot_dict


# Import chameli_logger lazily to avoid circular import
def get_chameli_logger():
    """Get chameli_logger instance to avoid circular imports."""
    from . import chameli_logger

    return chameli_logger


# Exception handler
def my_handler(typ, value, trace):
    get_chameli_logger().log_error(
        f"Unhandled exception: {typ.__name__} {value}",
        None,
        {
            "exception_type": typ.__name__,
            "exception_value": str(value),
            "traceback": "".join(traceback.format_tb(trace)),
        },
    )


def get_dynamic_config():
    return get_config()


sys.excepthook = my_handler

pd.options.display.float_format = "{:.2f}".format


def BlackScholesPrice(S: float, X: float, r: float, sigma: float, T: float, OptionType: str) -> float:
    """
    Calculate the price of a European option using the Black-Scholes formula.

    This function computes the theoretical price of a European call or put option
    based on the Black-Scholes model.

        S (float): The current spot price of the underlying asset.
        X (float): The strike price of the option.
        r (float): The risk-free interest rate (annualized).
        sigma (float): The volatility of the underlying asset (annualized).
        T (float): The time to maturity of the option (in years).
        OptionType (str): The type of the option. Use "C" for a call option and "P" for a put option.

        float: The calculated price of the option. Returns NaN if the input volatility
        is invalid (e.g., non-positive) or if the option type is unrecognized.

    Raises:
        ValueError: If any of the input parameters are invalid (e.g., negative time to maturity).

    Notes:
        - The function assumes continuous compounding for the risk-free rate.
        - The cumulative distribution function (CDF) of the standard normal distribution
          is used for calculations.
    """
    if math.isnan(sigma) or sigma <= 0:
        return float("nan")

    d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if OptionType[0] == "C":  # Call option
        bsp = S * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)
    elif OptionType[0] == "P":  # Put option
        bsp = X * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        bsp = float("nan")
    return bsp


def get_option_price(
    long_symbol: str,
    S: float,
    sigma: float,
    calc_time: str,
    r: float = 0,
    days_in_year=252,
    exchange="NSE",
):
    """
    Calculate the price of a European option using the Black-Scholes model.
    Args:
        long_symbol (str): The option symbol containing details such as expiry, type, and strike price.
                          Format: <underlying>_<expiry>_<type>_<strike_price>.
        S (float): Current price of the underlying asset.
        sigma (float): Volatility of the underlying asset (annualized).
        calc_time (str): The calculation time in string format (e.g., "YYYY-MM-DD HH:MM:SS").
        r (float, optional): Risk-free interest rate (annualized). Defaults to 0.
        days_in_year (int, optional): Number of trading days in a year. Defaults to 252.
        exchange (str, optional): The exchange where the option is traded (e.g., "NSE"). Defaults to "NSE".
    Returns:
        float: The calculated price of the European option.
    Raises:
        ValueError: If the `long_symbol` format is invalid or if `calc_time` is not a valid datetime.
    Notes:
        - The function assumes that the option is a European option.
        - The `valid_datetime` and `calc_fractional_business_days` functions are used to validate and compute
          the time to expiry in fractional business days.
        - The `BlackScholesPrice` function is used to compute the option price based on the Black-Scholes model.
    """

    OptionType = long_symbol.split("_")[3]
    X = float(long_symbol.split("_")[4])
    calc_time_dt, _ = valid_datetime(calc_time)
    expiry_str = long_symbol.split("_")[2]
    expiry = dt.datetime.strptime(expiry_str + " 15:30:00", "%Y%m%d %H:%M:%S")
    t = calc_fractional_business_days(calc_time_dt, expiry, exchange)
    t = t / days_in_year
    price = BlackScholesPrice(S, X, r, sigma, t, OptionType)
    return price


def _parse_combo_symbol(combo_symbol):
    result = OrderedDict()
    if "?" not in combo_symbol:
        result[combo_symbol] = 1
        return result

    symbols = combo_symbol.split(":")

    for symbol in symbols:
        name, quantity = symbol.split("?")
        result[name] = int(quantity)

    return result


def BlackScholesDelta(S: float, X: float, r: float, sigma: float, T: float, OptionType: str) -> float:
    """Calculate Black Scholes Delta

    Args:
        S (float): S is the spot price of the underlying asset
        X (float): X is the strike price
        r (float): r is the risk-free interest rate
        sigma (float): sigma is the volatility of the underlying asset
        T (float): T is the time to maturity of the option
        OptionType (str): OptionType is the type of the option (0 for a call option and 1 for a put option)


    Returns:
        float: Black Scholes Delta
    """
    if math.isnan(sigma) or sigma <= 0:
        return float("nan")

    d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    if OptionType[0] == "C":  # Call option
        bsd = norm.cdf(d1)
    elif OptionType[0] == "P":  # Put option
        bsd = norm.cdf(d1) - 1
    else:
        bsd = float("nan")
    return bsd


def calc_delta(
    long_symbol: str,
    opt_price: float,
    underlying: float,
    time: dt.datetime = dt.datetime.now(),
    days_in_year: int = 252,
    risk_free_rate: float = 0,
    exchange="NSE",
) -> float:
    """Helper function to calculate delta from long_symbol

    Args:
        long_symbol (str): long_symbol
        opt_price (float): option price
        underlying (float): underlying price
        time (dt.datetime, optional): Calculation time. Defaults to dt.datetime.now().
        days_in_year (int, optional): Defaults to 252.
        risk_free_rate (float, optional): Defaults to 0.
        exchage(str, optional): exchange. Defaults to "NSE".

    Returns:
        float: delta of option
    """
    strike = float(long_symbol.split("_")[4])
    option_type = long_symbol.split("_")[3]
    expiry_str = long_symbol.split("_")[2]
    expiry = dt.datetime.strptime(expiry_str + " 15:30:00", "%Y%m%d %H:%M:%S")
    t = calc_fractional_business_days(dt.datetime.now(), expiry, exchange=exchange)
    t = t / days_in_year

    sigma = BlackScholesIV(underlying, strike, risk_free_rate, t, option_type, opt_price)
    delta = BlackScholesDelta(underlying, strike, risk_free_rate, sigma, t, option_type)
    return delta


def BlackScholesGamma(S: float, X: float, r: float, sigma: float, T: float) -> float:
    """Calculate Black Scholes Gamma

    Args:
        S (float): S is the spot price of the underlying asset
        X (float): X is the strike price
        r (float): r is the risk-free interest rate
        sigma (float): sigma is the volatility of the underlying asset
        T (float): T is the time to maturity of the option

    Returns:
        float: Black Scholes Gamma
    """
    # Validate inputs
    if S <= 0 or X <= 0 or T <= 0 or math.isnan(sigma) or sigma <= 0:
        return float("nan")

    # Calculate d1
    d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    # Gamma formula
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))

    return gamma


def calc_greeks(
    long_symbol: str,
    opt_price: float,
    underlying: float,
    calc_time: dt.datetime = dt.datetime.now(),
    greeks: list[str] = ["delta", "gamma", "theta", "vega"],
    days_in_year: int = 252,
    risk_free_rate: float = 0,
    exchange="NSE",
    cap_theta: bool = True,  # New parameter
) -> float:
    """Helper function to calculate greeks from long_symbol

    Args:
        long_symbol (str): long_symbol
        opt_price (float): option price
        underlying (float): underlying price
        calc_time (dt.datetime, optional): Calculation time. Defaults to dt.datetime.now().
        greeks (list[str], optional): List of greeks to calculate. Defaults to ["delta", "gamma", "theta", "vega"].
        days_in_year (int, optional): Defaults to 252.
        risk_free_rate (float, optional): Defaults to 0.
        exchange (str, optional): exchange. Defaults to "NSE".
        cap_theta (bool, optional): Whether to cap theta at option price. Defaults to True.

    Returns:
        dict: Dictionary containing calculated greeks
    """
    out = {
        "vol": float("nan"),
        "delta": float("nan"),
        "gamma": float("nan"),
        "theta": float("nan"),
        "vega": float("nan"),
    }

    strike = float(long_symbol.split("_")[4])
    option_type = long_symbol.split("_")[3]
    expiry_str = long_symbol.split("_")[2]
    expiry = dt.datetime.strptime(expiry_str + " 15:30:00", "%Y%m%d %H:%M:%S")
    calc_time_dt, _ = valid_datetime(calc_time)
    t = calc_fractional_business_days(calc_time_dt, expiry, exchange=exchange)
    t = t / days_in_year

    sigma = BlackScholesIV(underlying, strike, risk_free_rate, t, option_type, opt_price)
    out["vol"] = sigma

    if "delta" in greeks:
        out["delta"] = BlackScholesDelta(underlying, strike, risk_free_rate, sigma, t, option_type)
    if "gamma" in greeks:
        out["gamma"] = BlackScholesGamma(underlying, strike, risk_free_rate, sigma, t)
    if "theta" in greeks:
        theta = BlackScholesTheta(underlying, strike, risk_free_rate, sigma, t, option_type, days_in_year)
        # Cap vega at option price if requested
        if cap_theta:
            # For long options, theta should be negative and abs(theta) <= option price
            if theta < 0 and abs(theta) > opt_price:
                theta = -opt_price
            # For short options (if any), theta should be positive and <= option price
            elif theta > 0 and theta > opt_price:
                theta = opt_price
        out["theta"] = theta
    if "vega" in greeks:
        out["vega"] = BlackScholesVega(underlying, strike, risk_free_rate, sigma, t)

    return convert_to_dot_dict(out)


def BlackScholesVega(S: float, X: float, r: float, sigma: float, T: float) -> float:
    """Calculate Black Scholes Vega

    Args:
        S (float): S is the spot price of the underlying asset
        X (float): X is the strike price
        r (float): r is the risk-free interest rate
        sigma (float): sigma is the volatility of the underlying asset
        T (float): T is the time to maturity of the option

    Returns:
        float: Black Scholes Vega
    """
    # Validate inputs
    if S <= 0 or X <= 0 or T <= 0 or math.isnan(sigma) or sigma <= 0:
        return float("nan")

    # Calculate d1
    d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    # Vega formula
    vega = S * math.exp(-r * T) * norm.pdf(d1) * math.sqrt(T) / 100

    return vega


def BlackScholesTheta(
    S: float,
    X: float,
    r: float,
    sigma: float,
    T: float,
    OptionType: str,
    daysInYear: int = 252,
) -> float:
    """Calculate Black Scholes Theta

    Args:
        S (float): S is the spot price of the underlying asset
        X (float): X is the strike price
        r (float): r is the risk-free interest rate
        sigma (float): sigma is the volatility of the underlying asset
        T (float): T is the time to maturity of the option
        OptionType (str): OptionType is the type of the option ("C" for call, "P" for put)
        daysInYear (int): Number of days in a year for annualizing the result (default is 365)

    Returns:
        float: Black Scholes Theta
    """
    # Validate inputs
    if S <= 0 or X <= 0 or T <= 0 or math.isnan(sigma) or sigma <= 0:
        return float("nan")

    # Calculate d1 and d2
    d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Calculate normal PDF of d1
    ndashd = norm.pdf(d1)

    # Calculate Theta for Call and Put options
    if OptionType[0] == "C":  # Call option
        theta = -(S * sigma * ndashd / (2 * math.sqrt(T))) - (r * X * math.exp(-r * T) * norm.cdf(d2))
    elif OptionType[0] == "P":  # Put option
        theta = -(S * sigma * ndashd / (2 * math.sqrt(T))) + (r * X * math.exp(-r * T) * norm.cdf(-d2))
    else:
        return float("nan")

    # Adjust for daily Theta
    return theta / daysInYear


def BlackScholesIV(
    S: float,
    X: float,
    r: float,
    T: float,
    OptionType: str,
    OptionPrice: float,
    seed=0.2,
) -> float:
    """Calculate Black-Scholes Implied Volatility (IV) using Brent's Method (Failsafe)"""

    if S <= 0 or X <= 0 or T <= 0 or math.isnan(OptionPrice):
        get_chameli_logger().log_error(
            f"Invalid Inputs: underlying: {S}, strike: {X}, years to maturity: {T}, option type: {OptionType}, option Price: {OptionPrice}",
            None,
            {
                "underlying": S,
                "strike": X,
                "years_to_maturity": T,
                "option_type": OptionType,
                "option_price": OptionPrice,
            },
        )
        return float("nan")  # Invalid inputs

    # Sanity checks: If the option price is clearly incorrect, return NaN
    if OptionType[0] == "C" and S - X > OptionPrice:
        return 0
    if OptionType[0] == "P" and X - S > OptionPrice:
        return 0

    def black_scholes_price(sigma):
        """Returns Black-Scholes price given sigma (used in root finding)"""
        d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if OptionType[0] == "C":  # Call option
            return S * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2) - OptionPrice
        elif OptionType[0] == "P":  # Put option
            return X * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - OptionPrice
        return float("nan")

    try:
        # Use Brent's method to find IV (safe & reliable)
        iv = brentq(black_scholes_price, 1e-6, 5.0, xtol=1e-6)
    except ValueError:
        # If Brent fails, fallback to Newton-Raphson
        iv = seed
        for _ in range(100):  # Max 100 iterations
            d1 = (math.log(S / X) + (r + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T)  # Vega

            if abs(vega) < 1e-6:  # Prevent division by near-zero
                break

            price_diff = black_scholes_price(iv)
            iv -= price_diff / vega  # Newton-Raphson update

            if abs(price_diff) < 1e-6:
                return iv

    return iv  # Return final implied volatility


def generate_opt_simulation(
    symbols: list,
    quantities: list,
    target_date: dt.datetime,
    vol_shift,
    start_range,
    end_range,
    increment,
    market_data: dict,
    exchange="NSE",
):
    """
    Simulates the performance of an options portfolio under various underlying price scenarios
    and calculates key metrics such as delta, theta, and vega.

    Args:
        symbols (list): A list of option symbols in the portfolio. Each symbol should follow
            the format "UNDERLYING_EXPIRY_OPTIONTYPE_STRIKE".
        quantities (list): A list of quantities corresponding to each option symbol.
        target_date (datetime.datetime): The target date for the simulation.
        vol_shift (float): The percentage shift in implied volatility for the simulation.
        start_range (float): The starting range for the underlying price simulation.
        end_range (float): The ending range for the underlying price simulation.
        increment (float): The increment step for the underlying price simulation.
        market_data (dict): A dictionary containing market data with the following keys:
            - "entry_prices" (list): Entry prices for the options.
            - "option_prices" (list): Current market prices of the options.
            - "underlying_prices" (list): Current prices of the underlying assets.
        exchange (str, optional): The exchange for which business days are calculated.
            Defaults to "NSE".

    Returns:
        dict: A dictionary containing the following keys:
            - "simulation" (pd.DataFrame): A DataFrame with columns:
                - "underlying": Simulated underlying prices.
                - "target_date_value": Portfolio value on the target date.
                - "maturity_value": Portfolio value at maturity.
            - "delta" (float): The portfolio's delta.
            - "theta" (float): The portfolio's theta.
            - "vega" (float): The portfolio's vega.
            - "underlying_prices" (list): The underlying prices used in the simulation.
            - "entry_prices" (list): The entry prices of the options.
            - "option_prices" (list): The current market prices of the options.
            - "init_value" (float): The initial value of the portfolio.
            - "mtm_value" (float): The mark-to-market value of the portfolio.
            - "symbols" (list): The list of option symbols.
            - "target_date" (datetime.datetime): The target date for the simulation.
            - "vol_shift" (float): The percentage shift in implied volatility.
            - "current_vols" (list): The current implied volatilities of the options.

    Raises:
        ValueError: If the "underlying_prices" key is missing or empty in the market_data dictionary.
    """

    current_vol: list[float] = []
    entry_prices: list[float] = []
    entry_prices_data = market_data.get("entry_prices")
    if entry_prices_data is None or len(entry_prices_data) == 0:
        entry_prices = market_data.get("option_prices", [])
    else:
        entry_prices = entry_prices_data

    # get minimum expiry for calendar spreads
    min_expiry = "29990101"
    for symbol in symbols:
        if min_expiry > symbol.split("_")[2]:
            min_expiry = symbol.split("_")[2]

    for symbol, price_u, price in zip(
        symbols, market_data.get("underlying_prices", []), market_data.get("option_prices", [])
    ):
        expiry = symbol.split("_")[2]
        expiry = dt.datetime.strptime(expiry + " 15:30:00", "%Y%m%d %H:%M:%S")
        t = calc_fractional_business_days(dt.datetime.now(), expiry, exchange=exchange)
        vol = BlackScholesIV(
            S=price_u,
            X=float(symbol.split("_")[4]),
            r=0,
            T=t / 252,
            OptionType=symbol.split("_")[3],
            OptionPrice=price,
        )
        current_vol.append(vol)
    structure_value_init = sum(list(map(lambda x, y: x * y, entry_prices, quantities)))
    structure_value_mtm = sum(list(map(lambda x, y: x * y, market_data.get("option_prices", []), quantities)))

    simulation_list = np.arange(
        int(start_range / increment) * increment - increment,
        int(end_range / increment) * increment + increment,
        step=increment,
    ).tolist()
    time_to_target = calc_fractional_business_days(dt.datetime.now(), target_date, exchange=exchange)
    bsp_value = []
    maturity_value = []
    underlying_prices = market_data.get("underlying_prices", [])
    if not underlying_prices:
        raise ValueError("Missing or empty 'underlying_prices' in market_data")

    underlying_ratios = [p / underlying_prices[0] for p in underlying_prices]

    for underlying in simulation_list:
        target_date_value = []
        expiry_value = []
        for symbol, vol, underlying_price, underlying_ratio, entry_price in zip(
            symbols,
            current_vol,
            underlying_prices,
            underlying_ratios,
            entry_prices,
        ):
            expiry = symbol.split("_")[2]
            strike = float(symbol.split("_")[4])
            t = (
                calc_fractional_business_days(
                    dt.datetime.now(),
                    dt.datetime.strptime(expiry + " 15:30:00", "%Y%m%d %H:%M:%S"),
                    exchange=exchange,
                )
                - time_to_target
            )
            bsp = BlackScholesPrice(
                S=underlying * underlying_ratio,
                X=strike,
                r=0,
                T=t / 252,
                OptionType=symbol.split("_")[3],
                sigma=vol * (1 + vol_shift / 100),
            )
            target_date_value.append(bsp - entry_price)
            if expiry == min_expiry:
                payoff = max(underlying - strike, 0) if symbol.split("_")[3] == "CALL" else max(strike - underlying, 0)
                payoff = payoff - entry_price
                expiry_value.append(payoff)
            else:
                time_to_min_expiry = calc_fractional_business_days(
                    dt.datetime.now(),
                    dt.datetime.strptime(min_expiry + " 15:30:00", "%Y%m%d %H:%M:%S"),
                    exchange=exchange,
                )
                t = (
                    calc_fractional_business_days(
                        dt.datetime.now(),
                        dt.datetime.strptime(expiry + " 15:30:00", "%Y%m%d %H:%M:%S"),
                        exchange=exchange,
                    )
                    - time_to_min_expiry
                )
                payoff = BlackScholesPrice(
                    S=underlying,
                    X=strike,
                    r=0,
                    T=t / 252,
                    OptionType=symbol.split("_")[3],
                    sigma=vol * (1 + vol_shift / 100),
                )
                payoff = payoff - entry_price
                expiry_value.append(payoff)
        bsp_value.append(sum(list(map(lambda x, y: x * y, target_date_value, quantities))))
        maturity_value.append(sum(list(map(lambda x, y: x * y, expiry_value, quantities))))
    delta_inr = []
    vega_inr = []
    theta_inr = []
    for symbol, vol, underlying_price, entry_price in zip(symbols, current_vol, underlying_prices, entry_prices):
        expiry = symbol.split("_")[2]
        strike = float(symbol.split("_")[4])
        t = calc_fractional_business_days(
            dt.datetime.now(),
            dt.datetime.strptime(expiry + " 15:30:00", "%Y%m%d %H:%M:%S"),
            exchange=exchange,
        )
        temp_delta = BlackScholesDelta(
            S=underlying_price,
            X=strike,
            r=0,
            sigma=vol,
            T=t / 252,
            OptionType=symbol.split("_")[3],
        )
        temp_vega = BlackScholesVega(S=underlying_price, X=strike, r=0, sigma=vol, T=t / 252)
        temp_theta = BlackScholesTheta(
            S=underlying_price,
            X=strike,
            r=0,
            sigma=vol,
            T=t / 252,
            OptionType=symbol.split("_")[3],
            daysInYear=252,
        )

        delta_inr.append(temp_delta)
        vega_inr.append(temp_vega)
        theta_inr.append(temp_theta)
        delta = sum(list(map(lambda x, y: x * y, delta_inr, quantities)))
        vega = sum(list(map(lambda x, y: x * y, vega_inr, quantities)))
        theta = sum(list(map(lambda x, y: x * y, theta_inr, quantities)))
    return {
        "simulation": pd.DataFrame(
            list(zip(simulation_list, bsp_value, maturity_value)),
            columns=["underlying", "target_date_value", "maturity_value"],
        ),
        "delta": delta,
        "theta": theta,
        "vega": vega,
        "underlying_prices": market_data.get("underlying_prices"),
        "entry_prices": entry_prices,
        "option_prices": market_data.get("option_prices"),
        "init_value": structure_value_init,
        "mtm_value": structure_value_mtm,
        "symbols": symbols,
        "target_date": target_date,
        "vol_shift": vol_shift,
        "current_vols": current_vol,
    }


def find_x_intercepts(price: list[float], value: list[float]) -> list[float]:
    """
    Find the x-intercepts (roots) of a function represented by discrete points.
    This function takes two lists: one representing the x-coordinates (price)
    and the other representing the corresponding y-coordinates (value). It
    returns a list of x-intercepts where the function crosses the x-axis.
        price (list[float]): List of x-coordinates (e.g., underlying prices).
        value (list[float]): List of y-coordinates (e.g., option values).
        list[float]: List of x-intercepts where the function crosses the x-axis.
    """
    x_intercepts = []
    for i in range(1, len(value)):
        if value[i] == 0:
            x_intercepts.append(value[i])
        elif (value[i - 1] > 0 and value[i] < 0) or (value[i - 1] < 0 and value[i] > 0):
            vp = [value[i - 1], value[i]]
            pp = [price[i - 1], price[i]]
            value_change = (vp[1] - vp[0]) / (pp[1] - pp[0])
            x_intercept = pp[0] + abs(vp[0] / value_change)
            x_intercepts.append(x_intercept)

    return x_intercepts


def performance_attribution(
    combo_symbol: str,
    spot_t0: float,
    spot_t1: float,
    ivs_t0: dict,
    ivs_t1: dict,
    t0: dt.datetime,
    t1: dt.datetime,
    r: float = 0,
    exchange: str = "NSE",
    market_close_time: str = "15:30:00",
    expiry_cutoff_minutes: int = 15,
    purchase_expiry_vol: float = 0.05,
) -> dict:
    """
    Attribution of price change for any option combo (long/short, any legs).
    ivs_t0 and ivs_t1 are dicts mapping symbol to IV at t0 and t1.
    Returns dict with total attribution and per-leg breakdown.



    New Attribution methodology (per user):
    1. Underlying change: P0 = price at exit time, entry spot, entry vol; P1 = price at exit time, exit spot, entry vol; attribution = P1 - P0
    2. Vol change: P0 = price at exit time, exit spot, entry vol; P1 = price at exit time, exit spot, exit vol; attribution = P1 - P0
    3. Time decay: P0 = price at entry time, entry spot, entry vol; P1 = price at exit time, entry spot, entry vol; attribution = P1 - P0
    Also determines if combo is purchase (net premium paid) or sale (net premium received), and applies special vol handling on expiry as described.
    Parameters:
        market_close_time (str): Market close time in HH:MM:SS (default '15:30:00')
        expiry_cutoff_minutes (int): Minutes before market close for sale vol rule (default 15)
        purchase_expiry_vol (float): Vol to use for purchase combos at/after market close on expiry (default 0.05)
    """
    t0 = apply_timezone(t0, exchange)
    t1 = apply_timezone(t1, exchange)
    legs = _parse_combo_symbol(combo_symbol)

    def get_intrinsic_value(symbol, spot_price):
        try:
            strike, option_type = parse_option_symbol(symbol)
            if option_type.upper() in ["CE", "CALL"]:
                return max(0, spot_price - strike)
            else:  # PUT
                return max(0, strike - spot_price)
        except Exception:
            return 0

    def get_expiry_datetime(symbol):
        try:
            expiry_str = symbol.split("_")[2]
            return dt.datetime.strptime(expiry_str + f" {market_close_time}", "%Y%m%d %H:%M:%S")
        except Exception:
            return None

    def is_on_expiry_and_after_cutoff(symbol, t1, exchange):
        expiry_dt = get_expiry_datetime(symbol)
        if expiry_dt is None:
            return False
        expiry_dt = apply_timezone(expiry_dt, exchange)
        # Calculate cutoff time
        cutoff_dt = expiry_dt - dt.timedelta(minutes=expiry_cutoff_minutes)
        return t1.date() == expiry_dt.date() and t1 >= cutoff_dt

    def is_on_expiry_and_after_close(symbol, t1):
        expiry_dt = get_expiry_datetime(symbol)
        if expiry_dt is None:
            return False
        return t1.date() == expiry_dt.date() and t1 >= expiry_dt

    def price(symbol, S, sigma, calc_time):
        sigma = max(sigma, 0.001)
        try:
            return get_option_price(
                long_symbol=symbol,
                S=S,
                sigma=sigma,
                calc_time=calc_time.strftime("%Y-%m-%d %H:%M:%S"),
                r=r,
                exchange=exchange,
            )
        except Exception:
            return get_intrinsic_value(symbol, S)

    entry_prices = {}
    for symbol, qty in legs.items():
        iv_t0 = max(ivs_t0.get(symbol, 0), 0.001)
        entry_prices[symbol] = price(symbol, spot_t0, iv_t0, t0)
    net_premium = sum(qty * entry_prices[symbol] for symbol, qty in legs.items())
    combo_type = "purchase" if net_premium > 0 else "sale"

    results = {}
    total = {
        k: 0.0
        for k in [
            "t0",
            "t1",
            "underlying_scenario_p0",
            "underlying_scenario_p1",
            "vol_scenario_p0",
            "vol_scenario_p1",
            "timedecay_scenario_p0",
            "timedecay_scenario_p1",
        ]
    }
    for symbol, qty in legs.items():
        iv_t0 = max(ivs_t0.get(symbol, 0), 0.001)
        iv_t1 = max(ivs_t1.get(symbol, 0), 0.001)
        expiry_dt = get_expiry_datetime(symbol)
        exit_iv = iv_t1
        if combo_type == "sale" and is_on_expiry_and_after_cutoff(symbol, t1, exchange):
            exit_iv = iv_t0
        elif combo_type == "purchase" and is_on_expiry_and_after_close(symbol, t1):
            exit_iv = purchase_expiry_vol
        p_t0 = price(symbol, spot_t0, iv_t0, t0)
        p_t1 = price(symbol, spot_t1, exit_iv, t1)
        underlying_scenario_p0 = price(symbol, spot_t0, iv_t0, t1)
        underlying_scenario_p1 = price(symbol, spot_t1, iv_t0, t1)
        vol_scenario_p0 = price(symbol, spot_t1, iv_t0, t1)
        vol_scenario_p1 = price(symbol, spot_t1, exit_iv, t1)
        timedecay_scenario_p0 = price(symbol, spot_t0, iv_t0, t0)
        timedecay_scenario_p1 = price(symbol, spot_t0, iv_t0, t1)
        results[symbol] = {
            "qty": qty,
            "p_t0": p_t0,
            "iv_t0": iv_t0,
            "t1": t1,
            "p_t1": p_t1,
            "iv_t1": exit_iv,
            "underlying_scenario_p0": underlying_scenario_p0,
            "underlying_scenario_p1": underlying_scenario_p1,
            "vol_scenario_p0": vol_scenario_p0,
            "vol_scenario_p1": vol_scenario_p1,
            "timedecay_scenario_p0": timedecay_scenario_p0,
            "timedecay_scenario_p1": timedecay_scenario_p1,
        }
        total["t0"] += qty * p_t0
        total["t1"] += qty * p_t1
        total["underlying_scenario_p0"] += qty * underlying_scenario_p0
        total["underlying_scenario_p1"] += qty * underlying_scenario_p1
        total["vol_scenario_p0"] += qty * vol_scenario_p0
        total["vol_scenario_p1"] += qty * vol_scenario_p1
        total["timedecay_scenario_p0"] += qty * timedecay_scenario_p0
        total["timedecay_scenario_p1"] += qty * timedecay_scenario_p1

    total_change = total["t1"] - total["t0"]
    underlying_attrib = total["underlying_scenario_p1"] - total["underlying_scenario_p0"]
    vol_attrib = total["vol_scenario_p1"] - total["vol_scenario_p0"]
    timedecay_attrib = total["timedecay_scenario_p1"] - total["timedecay_scenario_p0"]
    explained = underlying_attrib + vol_attrib + timedecay_attrib
    residual = total_change - explained

    return {
        "symbol": combo_symbol,
        "t0": t0,
        "t1": t1,
        "spot_t0": spot_t0,
        "spot_t1": spot_t1,
        "value_t0": total["t0"],
        "value_t1": total["t1"],
        "total_change": total_change,
        "underlying_attrib": underlying_attrib,
        "vol_attrib": vol_attrib,
        "timedecay_attrib": timedecay_attrib,
        "explained": explained,
        "residual": residual,
        "per_leg": results,
        "combo_type": combo_type,
        "net_premium": net_premium,
    }


# Helper functions you'll need to implement based on your system:


def parse_option_symbol(symbol):
    """
    Parse option symbol to extract strike and option type.
    Returns: (strike_price, option_type)
    Example: "NIFTY24DEC25000CE" -> (25000, "CE")
    """
    # Implement based on your symbol format
    return float(symbol.split("_")[4]), symbol.split("_")[3]


def get_expiry_time(symbol):
    """
    Get expiry datetime for an option symbol.
    Returns: datetime object
    """
    # Implement based on your system
    return dt.datetime.strptime(symbol.split("_")[2] + " 15:30:00", "%Y%m%d %H:%M:%S")
