from datetime import datetime

import pytest

from vnpy.trader.constant import Direction, Exchange, Interval, Status
from vnpy.trader.object import BarData, TickData

from vnpy_spreadtrading.backtesting import BacktestingEngine
from vnpy_spreadtrading.base import BacktestingMode, LegData, SpreadData
from vnpy_spreadtrading.template import SpreadAlgoTemplate, SpreadStrategyTemplate


class BarLaunchStrategy(SpreadStrategyTemplate):
    def __init__(
        self,
        strategy_engine: BacktestingEngine,
        strategy_name: str,
        spread: SpreadData,
        setting: dict[str, Direction | float],
    ) -> None:
        super().__init__(strategy_engine, strategy_name, spread, setting)
        self.launched: bool = False
        direction: Direction | float = setting["direction"]
        if not isinstance(direction, Direction):
            raise TypeError("direction must be Direction")
        self.launch_direction: Direction = direction
        self.launch_price: float = float(setting["price"])
        self.launch_volume: float = float(setting["volume"])

    def on_init(self) -> None:
        return

    def on_start(self) -> None:
        return

    def on_stop(self) -> None:
        return

    def on_spread_data(self) -> None:
        return

    def on_spread_tick(self, tick: TickData) -> None:
        return

    def on_spread_bar(self, bar: BarData) -> None:
        if self.launched:
            return
        self.launched = True
        if self.launch_direction == Direction.LONG:
            self.start_long_algo(self.launch_price, self.launch_volume, 0, 0)
            return
        self.start_short_algo(self.launch_price, self.launch_volume, 0, 0)

    def on_spread_pos(self) -> None:
        return

    def on_spread_algo(self, algo: SpreadAlgoTemplate) -> None:
        return


def make_leg(vt_symbol: str) -> LegData:
    leg: LegData = LegData(vt_symbol)
    leg.min_volume = 1
    leg.pricetick = 1
    return leg


def make_spread(leg: LegData) -> SpreadData:
    return SpreadData(
        "spread",
        [leg],
        {"leg0": leg.vt_symbol},
        {"leg0": 1},
        "leg0",
        {leg.vt_symbol: 1},
        leg.vt_symbol,
        1,
    )


def make_bar(when: datetime, close_price: float) -> BarData:
    return BarData(
        gateway_name="BACKTESTING",
        symbol="spread",
        exchange=Exchange.LOCAL,
        datetime=when,
        interval=Interval.MINUTE,
        open_price=close_price,
        high_price=close_price,
        low_price=close_price,
        close_price=close_price,
    )


def make_engine(
    direction: Direction,
    price: float,
    volume: float,
) -> BacktestingEngine:
    engine: BacktestingEngine = BacktestingEngine()
    engine.set_parameters(
        spread=make_spread(make_leg("leg.LOCAL")),
        interval=Interval.MINUTE,
        start=datetime(2024, 1, 2, 9, 0),
        rate=0,
        slippage=0,
        size=10,
        pricetick=1,
        capital=1_000_000,
        end=datetime(2024, 1, 3, 15, 0),
        mode=BacktestingMode.BAR,
    )
    engine.add_strategy(
        BarLaunchStrategy,
        {
            "direction": direction,
            "price": price,
            "volume": volume,
        },
    )
    engine.strategy.trading = True
    return engine


@pytest.mark.parametrize(
    ("direction", "bar_close", "pos_change"),
    [
        (Direction.LONG, 99.0, 3.0),
        (Direction.SHORT, 101.0, -3.0),
    ],
)
def test_later_bar_advances_clock_and_fills_crossed_algo(
    direction: Direction,
    bar_close: float,
    pos_change: float,
) -> None:
    engine: BacktestingEngine = make_engine(direction, 100, 3)
    first: datetime = datetime(2024, 1, 2, 9, 0)
    second: datetime = datetime(2024, 1, 2, 9, 1)
    next_day: datetime = datetime(2024, 1, 3, 9, 0)

    engine.new_bar(make_bar(first, bar_close))

    assert engine.datetime == first
    assert set(engine.active_algos) == {"1"}
    assert engine.algos["1"].status == Status.NOTTRADED
    assert engine.algos["1"].traded == 0
    assert engine.algos["1"].traded_volume == 0
    assert engine.spread.net_pos == 0
    assert engine.trades == {}
    assert engine.daily_results[first.date()].close_price == bar_close

    engine.new_bar(make_bar(second, bar_close))

    filled: SpreadAlgoTemplate = engine.algos["1"]
    assert engine.datetime == second
    assert second > first
    assert engine.active_algos == {}
    assert filled.status == Status.ALLTRADED
    assert filled.traded == pos_change
    assert filled.traded_volume == 3
    assert filled.traded_price == 100
    assert engine.spread.net_pos == pos_change
    assert engine.strategy.algoids == set()
    trade = engine.trades["BACKTESTING.1"]
    assert trade.direction == direction
    assert trade.volume == 3
    assert trade.price == bar_close
    assert trade.datetime == second
    assert engine.daily_results[first.date()].close_price == bar_close

    engine.new_bar(make_bar(next_day, 90))

    assert engine.datetime == next_day
    assert engine.algo_count == 1
    assert len(engine.algos) == 1
    assert engine.spread.net_pos == pos_change
    assert engine.daily_results[first.date()].close_price == bar_close
    assert engine.daily_results[next_day.date()].close_price == 90


def test_uncrossed_bars_advance_datetime_and_keep_algo_active() -> None:
    engine: BacktestingEngine = make_engine(Direction.LONG, 100, 2)
    first: datetime = datetime(2024, 1, 2, 9, 0)
    second: datetime = datetime(2024, 1, 2, 9, 1)
    next_day: datetime = datetime(2024, 1, 3, 9, 5)

    engine.new_bar(make_bar(first, 110))
    assert engine.datetime == first
    assert engine.algos["1"].status == Status.NOTTRADED
    assert engine.daily_results[first.date()].close_price == 110

    engine.new_bar(make_bar(second, 120))

    assert engine.datetime == second
    assert second > first
    assert set(engine.active_algos) == {"1"}
    assert engine.algos["1"].status == Status.NOTTRADED
    assert engine.algos["1"].traded == 0
    assert engine.algos["1"].traded_volume == 0
    assert engine.spread.net_pos == 0
    assert engine.trades == {}
    assert engine.daily_results[first.date()].close_price == 120

    engine.new_bar(make_bar(next_day, 130))

    assert engine.datetime == next_day
    assert set(engine.active_algos) == {"1"}
    assert engine.algos["1"].traded_volume == 0
    assert engine.daily_results[first.date()].close_price == 120
    assert engine.daily_results[next_day.date()].close_price == 130
