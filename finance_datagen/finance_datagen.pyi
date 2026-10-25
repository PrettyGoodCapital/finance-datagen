from collections.abc import Sequence

from pyarrow import RecordBatch

class GBMGenerator:
    def __init__(
        self,
        s0: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        dt: float = ...,
        n_steps: int = 252,
        symbol: str = "SYM",
        start_ms: int = 0,
        step_ms: int = 86400000,
        seed: int | None = None,
    ) -> None: ...
    def record_batch(self) -> RecordBatch: ...

class HestonGenerator:
    def __init__(
        self,
        s0: float = 100.0,
        v0: float = 0.04,
        mu: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        dt: float = ...,
        n_steps: int = 252,
        symbol: str = "SYM",
        start_ms: int = 0,
        step_ms: int = 86400000,
        seed: int | None = None,
    ) -> None: ...
    def record_batch(self) -> RecordBatch: ...

class GARCHGenerator:
    def __init__(
        self,
        s0: float = 100.0,
        mu: float = 0.0,
        omega: float = 1e-6,
        alpha: float = 0.05,
        beta: float = 0.90,
        n_steps: int = 252,
        symbol: str = "SYM",
        start_ms: int = 0,
        step_ms: int = 86400000,
        seed: int | None = None,
    ) -> None: ...
    def record_batch(self) -> RecordBatch: ...

def ohlc_from_close(
    close: Sequence[float],
    intrabar_vol: float = 0.005,
    base_volume: float = 1000000.0,
    vol_factor: float = 5e7,
    symbol: str = "SYM",
    start_ms: int = 0,
    step_ms: int = 86400000,
    seed: int | None = None,
) -> RecordBatch: ...
