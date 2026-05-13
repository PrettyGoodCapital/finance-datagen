"""Synthetic positions and transactions for post-trade workflows."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

import numpy as np
import polars as pl
from finance_enums import PositionEffect, Side
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

    @model_validator(mode="after")
    def _validate_symbols(self):
        _symbols(self.n_assets, self.symbols)
        return self

    def generate(self) -> pl.DataFrame:
        """Return ``[date, symbol, price, quantity, market_value, weight]``."""
        rng = np.random.default_rng(self.seed)
        dates = _date_range(self.n_dates, self.start)
        symbols = _symbols(self.n_assets, self.symbols)

        base_prices = self.average_price * rng.lognormal(mean=0.0, sigma=0.25, size=self.n_assets)
        returns = rng.normal(0.0002, self.price_vol, size=(self.n_dates, self.n_assets))
        prices = base_prices * np.exp(np.cumsum(returns, axis=0))

        raw_weights = rng.normal(0.0, 1.0, size=(self.n_dates, self.n_assets))
        raw_weights /= np.abs(raw_weights).sum(axis=1, keepdims=True)
        weights = raw_weights * self.gross_exposure
        market_values = weights * self.portfolio_value
        quantities = market_values / prices

        return pl.DataFrame(
            {
                "date": np.repeat(np.array(dates, dtype="datetime64[D]"), self.n_assets),
                "symbol": np.tile(np.array(symbols), self.n_dates),
                "price": prices.reshape(-1),
                "quantity": quantities.reshape(-1),
                "market_value": market_values.reshape(-1),
                "weight": weights.reshape(-1),
            }
        ).with_columns(pl.col("date").cast(pl.Date))


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

    @model_validator(mode="after")
    def _validate_symbols(self):
        _symbols(self.n_assets, self.symbols)
        return self

    def generate(self) -> pl.DataFrame:
        """Return transaction rows with side labels and explicit costs."""
        rng = np.random.default_rng(self.seed)
        dates = _date_range(self.n_dates, self.start)
        n_rows = self.n_dates * self.trades_per_day
        transaction_intents = np.asarray(_TRANSACTION_INTENTS, dtype=object)
        symbols = _symbols(self.n_assets, self.symbols)

        timestamps: list[datetime] = []
        for current_date in dates:
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

        return pl.DataFrame(
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
    ).generate()
