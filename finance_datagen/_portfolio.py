"""Synthetic positions and transactions for post-trade workflows."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

import numpy as np
import polars as pl
from finance_dates import Calendar
from finance_enums import (
    Currency,
    ExchangeCode,
    OrderStatus,
    OrderType,
    PositionEffect,
    Side,
    TimeInForce,
    exchange_record,
)
from pydantic import model_validator

from ._base import DataGenerator, NonNegativeFloat, PositiveFloat, PositiveInt

_TRANSACTION_INTENTS = (
    (Side.Buy.value, PositionEffect.Open.value),
    (Side.Sell.value, PositionEffect.Close.value),
    (Side.Sell.value, PositionEffect.Open.value),
    (Side.Buy.value, PositionEffect.Close.value),
)


def _date_range(n_dates: int, start: date | None) -> list[date]:
    first = date(2020, 1, 1) if start is None else start
    return [first + timedelta(days=i) for i in range(n_dates)]


def _trading_date_range(n_dates: int, start: date | None, exchange: str | None) -> list[date]:
    if exchange is None:
        return _date_range(n_dates, start)

    first = date(2020, 1, 1) if start is None else start
    calendar = Calendar.from_exchange(exchange)
    window_end = first + timedelta(days=max(10, n_dates * 3))
    dates = calendar.business_days(first, window_end)
    while len(dates) < n_dates:
        window_end = window_end + timedelta(days=max(10, n_dates * 2))
        dates = calendar.business_days(first, window_end)
    return dates[:n_dates]


def _symbols(n_assets: int, symbols: Sequence[str] | None) -> list[str]:
    values = [f"A{i:04d}" for i in range(n_assets)] if symbols is None else list(symbols)
    if len(values) != n_assets:
        raise ValueError(f"symbols length {len(values)} != n_assets {n_assets}")
    return values


class PositionsGenerator(DataGenerator[pl.DataFrame]):
    """Generate a long-form synthetic positions table."""

    n_dates: PositiveInt = 252
    n_assets: PositiveInt = 50
    portfolio_value: PositiveFloat = 1_000_000.0
    gross_exposure: PositiveFloat = 1.0
    average_price: PositiveFloat = 100.0
    price_vol: NonNegativeFloat = 0.02
    seed: int | None = None
    start: date | None = None
    symbols: tuple[str, ...] | None = None
    currency: str | None = None
    exchange: str | None = None
    include_region: bool = False

    @model_validator(mode="after")
    def _validate_symbols(self):
        _symbols(self.n_assets, self.symbols)
        if self.currency is not None:
            Currency(self.currency)
        if self.exchange is not None:
            ExchangeCode(self.exchange)
        return self

    def generate(self) -> pl.DataFrame:
        """Return ``[date, symbol, price, quantity, market_value, weight]``."""
        rng = np.random.default_rng(self.seed)
        dates = _trading_date_range(self.n_dates, self.start, self.exchange)
        symbols = _symbols(self.n_assets, self.symbols)

        base_prices = self.average_price * rng.lognormal(mean=0.0, sigma=0.25, size=self.n_assets)
        returns = rng.normal(0.0002, self.price_vol, size=(self.n_dates, self.n_assets))
        prices = base_prices * np.exp(np.cumsum(returns, axis=0))

        raw_weights = rng.normal(0.0, 1.0, size=(self.n_dates, self.n_assets))
        raw_weights /= np.abs(raw_weights).sum(axis=1, keepdims=True)
        weights = raw_weights * self.gross_exposure
        market_values = weights * self.portfolio_value
        quantities = market_values / prices

        frame = pl.DataFrame(
            {
                "date": np.repeat(np.array(dates, dtype="datetime64[D]"), self.n_assets),
                "symbol": np.tile(np.array(symbols), self.n_dates),
                "price": prices.reshape(-1),
                "quantity": quantities.reshape(-1),
                "market_value": market_values.reshape(-1),
                "weight": weights.reshape(-1),
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        if self.currency is not None:
            frame = frame.with_columns(pl.lit(self.currency).alias("currency"))

        if self.exchange is not None:
            frame = frame.with_columns(pl.lit(self.exchange).alias("exchange"))
            if self.include_region:
                record = exchange_record(self.exchange)
                region = None if record is None else record.region
                frame = frame.with_columns(pl.lit(region).alias("region"))

        return frame


class TransactionsGenerator(DataGenerator[pl.DataFrame]):
    """Generate a synthetic transaction log for post-trade tests."""

    n_dates: PositiveInt = 252
    n_assets: PositiveInt = 50
    trades_per_day: PositiveInt = 25
    average_price: PositiveFloat = 100.0
    price_vol: NonNegativeFloat = 0.25
    max_amount: PositiveInt = 1_000
    commission: NonNegativeFloat = 1.0
    fee_bps: NonNegativeFloat = 0.2
    bps: NonNegativeFloat = 5.0
    seed: int | None = None
    start: date | None = None
    symbols: tuple[str, ...] | None = None
    currency: str | None = None
    exchange: str | None = None
    include_region: bool = False

    @model_validator(mode="after")
    def _validate_symbols(self):
        _symbols(self.n_assets, self.symbols)
        if self.currency is not None:
            Currency(self.currency)
        if self.exchange is not None:
            ExchangeCode(self.exchange)
        return self

    def generate(self) -> pl.DataFrame:
        """Return transaction rows with side labels and explicit costs."""
        rng = np.random.default_rng(self.seed)
        dates = _trading_date_range(self.n_dates, self.start, self.exchange)
        n_rows = self.n_dates * self.trades_per_day
        transaction_intents = np.asarray(_TRANSACTION_INTENTS, dtype=object)
        symbols = _symbols(self.n_assets, self.symbols)

        timestamps: list[datetime] = []
        calendar = Calendar.from_exchange(self.exchange) if self.exchange is not None else None
        for current_date in dates:
            if calendar is not None:
                sessions = calendar.sessions(current_date, current_date)
                if sessions:
                    session_open, session_close = sessions[0]
                    span_seconds = max(1, int((session_close - session_open).total_seconds()))
                    offsets = np.sort(rng.integers(0, span_seconds + 1, size=self.trades_per_day))
                    timestamps.extend(session_open + timedelta(seconds=int(offset)) for offset in offsets)
                    continue
            base = datetime.combine(current_date, time(9, 30), tzinfo=timezone.utc)
            offsets = np.sort(rng.integers(0, 6 * 60 * 60 + 30 * 60, size=self.trades_per_day))
            timestamps.extend(base + timedelta(seconds=int(offset)) for offset in offsets)

        intent_indices = rng.integers(0, len(_TRANSACTION_INTENTS), size=n_rows)
        side_values = transaction_intents[intent_indices, 0]
        position_effect_values = transaction_intents[intent_indices, 1]
        raw_amounts = rng.integers(1, self.max_amount + 1, size=n_rows).astype(float)
        signed_amounts = np.where(side_values == Side.Buy.value, raw_amounts, -raw_amounts)
        prices = self.average_price * rng.lognormal(mean=0.0, sigma=self.price_vol, size=n_rows)
        notional = np.abs(signed_amounts) * prices

        frame = pl.DataFrame(
            {
                "timestamp": pl.Series(timestamps).cast(pl.Datetime("ms", "UTC")),
                "symbol": rng.choice(np.array(symbols), size=n_rows),
                "amount": signed_amounts,
                "price": prices,
                "side": side_values,
                "position_effect": position_effect_values,
                "notional": notional,
                "commission": np.full(n_rows, self.commission),
                "fees": notional * self.fee_bps / 10_000.0,
                "bps": np.full(n_rows, self.bps),
            }
        )

        if self.currency is not None:
            frame = frame.with_columns(pl.lit(self.currency).alias("currency"))

        if self.exchange is not None:
            frame = frame.with_columns(pl.lit(self.exchange).alias("exchange"))
            if self.include_region:
                record = exchange_record(self.exchange)
                region = None if record is None else record.region
                frame = frame.with_columns(pl.lit(region).alias("region"))

        return frame


class OrdersGenerator(DataGenerator[pl.DataFrame]):
    """Generate enum-backed synthetic order fixtures."""

    n_dates: PositiveInt = 252
    n_assets: PositiveInt = 50
    orders_per_day: PositiveInt = 25
    average_price: PositiveFloat = 100.0
    price_vol: NonNegativeFloat = 0.2
    max_quantity: PositiveInt = 1_000
    seed: int | None = None
    start: date | None = None
    symbols: tuple[str, ...] | None = None
    exchange: str | None = None
    currency: str | None = None
    include_region: bool = False

    @model_validator(mode="after")
    def _validate_symbols(self):
        _symbols(self.n_assets, self.symbols)
        if self.exchange is not None:
            ExchangeCode(self.exchange)
        if self.currency is not None:
            Currency(self.currency)
        return self

    def generate(self) -> pl.DataFrame:
        """Return ``[timestamp, symbol, order_id, side, order_type, quantity, limit_price, order_status, time_in_force]``."""
        rng = np.random.default_rng(self.seed)
        dates = _trading_date_range(self.n_dates, self.start, self.exchange)
        symbols = _symbols(self.n_assets, self.symbols)
        n_rows = self.n_dates * self.orders_per_day

        calendar = Calendar.from_exchange(self.exchange) if self.exchange is not None else None
        timestamps: list[datetime] = []
        for current_date in dates:
            if calendar is not None:
                sessions = calendar.sessions(current_date, current_date)
                if sessions:
                    session_open, session_close = sessions[0]
                    span_seconds = max(1, int((session_close - session_open).total_seconds()))
                    offsets = np.sort(rng.integers(0, span_seconds + 1, size=self.orders_per_day))
                    timestamps.extend(session_open + timedelta(seconds=int(offset)) for offset in offsets)
                    continue
            base = datetime.combine(current_date, time(9, 30), tzinfo=timezone.utc)
            offsets = np.sort(rng.integers(0, 6 * 60 * 60 + 30 * 60, size=self.orders_per_day))
            timestamps.extend(base + timedelta(seconds=int(offset)) for offset in offsets)

        frame = pl.DataFrame(
            {
                "timestamp": pl.Series(timestamps).cast(pl.Datetime("ms", "UTC")),
                "symbol": rng.choice(np.array(symbols), size=n_rows),
                "order_id": [f"ORD-{i:08d}" for i in range(n_rows)],
                "side": rng.choice(np.array([Side.Buy.value, Side.Sell.value]), size=n_rows),
                "order_type": rng.choice(np.array([OrderType.Market.value, OrderType.Limit.value]), size=n_rows, p=[0.55, 0.45]),
                "quantity": rng.integers(1, self.max_quantity + 1, size=n_rows),
                "limit_price": self.average_price * rng.lognormal(mean=0.0, sigma=self.price_vol, size=n_rows),
                "order_status": rng.choice(
                    np.array(
                        [
                            OrderStatus.New.value,
                            OrderStatus.PartiallyFilled.value,
                            OrderStatus.Filled.value,
                            OrderStatus.Canceled.value,
                            OrderStatus.Rejected.value,
                        ]
                    ),
                    size=n_rows,
                    p=[0.35, 0.25, 0.25, 0.10, 0.05],
                ),
                "time_in_force": rng.choice(
                    np.array([TimeInForce.Day.value, TimeInForce.GoodTillCanceled.value]),
                    size=n_rows,
                    p=[0.80, 0.20],
                ),
            }
        )

        if self.currency is not None:
            frame = frame.with_columns(pl.lit(self.currency).alias("currency"))
        if self.exchange is not None:
            frame = frame.with_columns(pl.lit(self.exchange).alias("exchange"))
            if self.include_region:
                record = exchange_record(self.exchange)
                region = None if record is None else record.region
                frame = frame.with_columns(pl.lit(region).alias("region"))

        return frame


class ExecutionsGenerator(DataGenerator[pl.DataFrame]):
    """Generate synthetic execution fixtures tied to synthetic orders."""

    n_dates: PositiveInt = 252
    n_assets: PositiveInt = 50
    executions_per_day: PositiveInt = 30
    average_price: PositiveFloat = 100.0
    price_vol: NonNegativeFloat = 0.2
    max_quantity: PositiveInt = 1_000
    seed: int | None = None
    start: date | None = None
    symbols: tuple[str, ...] | None = None
    exchange: str | None = None
    currency: str | None = None
    include_region: bool = False

    @model_validator(mode="after")
    def _validate_symbols(self):
        _symbols(self.n_assets, self.symbols)
        if self.exchange is not None:
            ExchangeCode(self.exchange)
        if self.currency is not None:
            Currency(self.currency)
        return self

    def generate(self) -> pl.DataFrame:
        """Return ``[timestamp, order_id, symbol, side, price, quantity, liquidity_flag]``."""
        rng = np.random.default_rng(self.seed)
        dates = _trading_date_range(self.n_dates, self.start, self.exchange)
        symbols = _symbols(self.n_assets, self.symbols)
        n_rows = self.n_dates * self.executions_per_day

        calendar = Calendar.from_exchange(self.exchange) if self.exchange is not None else None
        timestamps: list[datetime] = []
        for current_date in dates:
            if calendar is not None:
                sessions = calendar.sessions(current_date, current_date)
                if sessions:
                    session_open, session_close = sessions[0]
                    span_seconds = max(1, int((session_close - session_open).total_seconds()))
                    offsets = np.sort(rng.integers(0, span_seconds + 1, size=self.executions_per_day))
                    timestamps.extend(session_open + timedelta(seconds=int(offset)) for offset in offsets)
                    continue
            base = datetime.combine(current_date, time(9, 30), tzinfo=timezone.utc)
            offsets = np.sort(rng.integers(0, 6 * 60 * 60 + 30 * 60, size=self.executions_per_day))
            timestamps.extend(base + timedelta(seconds=int(offset)) for offset in offsets)

        frame = pl.DataFrame(
            {
                "timestamp": pl.Series(timestamps).cast(pl.Datetime("ms", "UTC")),
                "execution_id": [f"EXE-{i:08d}" for i in range(n_rows)],
                "order_id": [f"ORD-{int(i / 2):08d}" for i in range(n_rows)],
                "symbol": rng.choice(np.array(symbols), size=n_rows),
                "side": rng.choice(np.array([Side.Buy.value, Side.Sell.value]), size=n_rows),
                "price": self.average_price * rng.lognormal(mean=0.0, sigma=self.price_vol, size=n_rows),
                "quantity": rng.integers(1, self.max_quantity + 1, size=n_rows),
                "liquidity_flag": rng.choice(np.array(["Added", "Removed", "Auction"]), size=n_rows, p=[0.4, 0.5, 0.1]),
                "time_in_force": rng.choice(
                    np.array([TimeInForce.Day.value, TimeInForce.GoodTillCanceled.value]),
                    size=n_rows,
                    p=[0.80, 0.20],
                ),
            }
        )

        if self.currency is not None:
            frame = frame.with_columns(pl.lit(self.currency).alias("currency"))
        if self.exchange is not None:
            frame = frame.with_columns(pl.lit(self.exchange).alias("exchange"))
            if self.include_region:
                record = exchange_record(self.exchange)
                region = None if record is None else record.region
                frame = frame.with_columns(pl.lit(region).alias("region"))

        return frame


def generate_positions(
    n_dates: int = 252,
    n_assets: int = 50,
    portfolio_value: float = 1_000_000.0,
    gross_exposure: float = 1.0,
    average_price: float = 100.0,
    price_vol: float = 0.02,
    seed: int | None = None,
    start: date | None = None,
    symbols: Sequence[str] | None = None,
    currency: str | None = None,
    exchange: str | None = None,
    include_region: bool = False,
) -> pl.DataFrame:
    """Generate a synthetic positions table."""
    return PositionsGenerator(
        n_dates=n_dates,
        n_assets=n_assets,
        portfolio_value=portfolio_value,
        gross_exposure=gross_exposure,
        average_price=average_price,
        price_vol=price_vol,
        seed=seed,
        start=start,
        symbols=None if symbols is None else tuple(symbols),
        currency=currency,
        exchange=exchange,
        include_region=include_region,
    ).generate()


def generate_transactions(
    n_dates: int = 252,
    n_assets: int = 50,
    trades_per_day: int = 25,
    average_price: float = 100.0,
    price_vol: float = 0.25,
    max_amount: int = 1_000,
    commission: float = 1.0,
    fee_bps: float = 0.2,
    bps: float = 5.0,
    seed: int | None = None,
    start: date | None = None,
    symbols: Sequence[str] | None = None,
    currency: str | None = None,
    exchange: str | None = None,
    include_region: bool = False,
) -> pl.DataFrame:
    """Generate a synthetic transaction log."""
    return TransactionsGenerator(
        n_dates=n_dates,
        n_assets=n_assets,
        trades_per_day=trades_per_day,
        average_price=average_price,
        price_vol=price_vol,
        max_amount=max_amount,
        commission=commission,
        fee_bps=fee_bps,
        bps=bps,
        seed=seed,
        start=start,
        symbols=None if symbols is None else tuple(symbols),
        currency=currency,
        exchange=exchange,
        include_region=include_region,
    ).generate()


def generate_orders(
    n_dates: int = 252,
    n_assets: int = 50,
    orders_per_day: int = 25,
    average_price: float = 100.0,
    price_vol: float = 0.2,
    max_quantity: int = 1_000,
    seed: int | None = None,
    start: date | None = None,
    symbols: Sequence[str] | None = None,
    currency: str | None = None,
    exchange: str | None = None,
    include_region: bool = False,
) -> pl.DataFrame:
    """Generate synthetic order fixtures."""
    return OrdersGenerator(
        n_dates=n_dates,
        n_assets=n_assets,
        orders_per_day=orders_per_day,
        average_price=average_price,
        price_vol=price_vol,
        max_quantity=max_quantity,
        seed=seed,
        start=start,
        symbols=None if symbols is None else tuple(symbols),
        currency=currency,
        exchange=exchange,
        include_region=include_region,
    ).generate()


def generate_executions(
    n_dates: int = 252,
    n_assets: int = 50,
    executions_per_day: int = 30,
    average_price: float = 100.0,
    price_vol: float = 0.2,
    max_quantity: int = 1_000,
    seed: int | None = None,
    start: date | None = None,
    symbols: Sequence[str] | None = None,
    currency: str | None = None,
    exchange: str | None = None,
    include_region: bool = False,
) -> pl.DataFrame:
    """Generate synthetic execution fixtures."""
    return ExecutionsGenerator(
        n_dates=n_dates,
        n_assets=n_assets,
        executions_per_day=executions_per_day,
        average_price=average_price,
        price_vol=price_vol,
        max_quantity=max_quantity,
        seed=seed,
        start=start,
        symbols=None if symbols is None else tuple(symbols),
        currency=currency,
        exchange=exchange,
        include_region=include_region,
    ).generate()
