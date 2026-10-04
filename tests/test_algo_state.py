from datetime import datetime

import pytest

from vnpy.trader.constant import Direction, Exchange, Product, Status
from vnpy.trader.object import ContractData, OrderData, TickData, TradeData

from vnpy_spreadtrading.algo import SpreadTakerAlgo
from vnpy_spreadtrading.base import LegData, SpreadData
from vnpy_spreadtrading.template import SpreadAlgoTemplate


class RecordingAlgoEngine:
    def __init__(self) -> None:
        self.sent: list[tuple[str, float, float, Direction]] = []
        self.cancelled: list[str] = []
        self.order_ids: list[str] = []
        self.ticks: dict[str, TickData] = {}
        self.contracts: dict[str, ContractData] = {}
        self._next_order: int = 0

    def write_algo_log(self, algo: SpreadAlgoTemplate, msg: str) -> None:
        return

    def put_algo_event(self, algo: SpreadAlgoTemplate) -> None:
        return

    def get_tick(self, vt_symbol: str) -> TickData | None:
        return self.ticks.get(vt_symbol)

    def get_contract(self, vt_symbol: str) -> ContractData | None:
        return self.contracts.get(vt_symbol)

    def send_order(
        self,
        algo: SpreadAlgoTemplate,
        vt_symbol: str,
        price: float,
        volume: float,
        direction: Direction,
        lock: bool,
        fak: bool,
    ) -> list[str]:
        self._next_order += 1
        vt_orderid: str = f"FAKE.{self._next_order}"
        self.order_ids.append(vt_orderid)
        self.sent.append((vt_symbol, price, volume, direction))
        return [vt_orderid]

    def cancel_order(self, algo: SpreadAlgoTemplate, vt_orderid: str) -> None:
        self.cancelled.append(vt_orderid)


def make_leg(vt_symbol: str, min_volume: float = 1) -> LegData:
    leg: LegData = LegData(vt_symbol)
    leg.min_volume = min_volume
    leg.pricetick = 1
    leg.bid_price = 10
    leg.ask_price = 11
    leg.bid_volume = 12
    leg.ask_volume = 9
    return leg


def make_spread(
    legs: list[LegData],
    trading_multipliers: dict[str, int],
    active_symbol: str,
    min_volume: float = 1,
) -> SpreadData:
    variable_symbols: dict[str, str] = {}
    variable_directions: dict[str, int] = {}
    formula_parts: list[str] = []

    for index, leg in enumerate(legs):
        variable: str = f"leg{index}"
        variable_symbols[variable] = leg.vt_symbol
        multiplier: int = trading_multipliers[leg.vt_symbol]
        variable_directions[variable] = 1 if multiplier >= 0 else -1
        formula_parts.append(variable)

    return SpreadData(
        "spread",
        legs,
        variable_symbols,
        variable_directions,
        "+".join(formula_parts),
        trading_multipliers,
        active_symbol,
        min_volume,
    )


def make_algo(
    spread: SpreadData,
    engine: RecordingAlgoEngine,
    direction: Direction = Direction.LONG,
    price: float = 0,
    volume: float = 1,
    payup: int = 0,
    interval: int = 0,
    algo_class: type[SpreadAlgoTemplate] = SpreadAlgoTemplate,
) -> SpreadAlgoTemplate:
    return algo_class(
        engine,
        "algo",
        spread,
        direction,
        price,
        volume,
        payup,
        interval,
        False,
        {},
    )


def make_contract(vt_symbol: str, min_volume: float = 1) -> ContractData:
    symbol, exchange_name = vt_symbol.split(".")
    return ContractData(
        gateway_name="FAKE",
        symbol=symbol,
        exchange=Exchange(exchange_name),
        name=symbol,
        product=Product.FUTURES,
        size=1,
        pricetick=1,
        min_volume=min_volume,
    )


def attach_order(
    algo: SpreadAlgoTemplate,
    vt_symbol: str,
    volume: float,
    orderid: str,
    direction: Direction,
) -> OrderData:
    symbol, exchange_name = vt_symbol.split(".")
    order: OrderData = OrderData(
        gateway_name="FAKE",
        symbol=symbol,
        exchange=Exchange(exchange_name),
        orderid=orderid,
        direction=direction,
        volume=volume,
        status=Status.NOTTRADED,
    )
    algo.orders[order.vt_orderid] = order
    algo.leg_orders[vt_symbol].append(order.vt_orderid)
    return order


def make_trade(
    order: OrderData,
    tradeid: str,
    volume: float,
    price: float,
    direction: Direction,
) -> TradeData:
    return TradeData(
        gateway_name=order.gateway_name,
        symbol=order.symbol,
        exchange=order.exchange,
        orderid=order.orderid,
        tradeid=tradeid,
        direction=direction,
        price=price,
        volume=volume,
    )


@pytest.mark.parametrize(
    ("direction", "trade_direction", "sign"),
    [
        (Direction.LONG, Direction.LONG, 1),
        (Direction.SHORT, Direction.SHORT, -1),
    ],
)
def test_fills_move_status_from_partial_to_alltraded_and_drop_active_order(
    direction: Direction,
    trade_direction: Direction,
    sign: int,
) -> None:
    leg: LegData = make_leg("leg.LOCAL")
    spread: SpreadData = make_spread(
        [leg],
        {leg.vt_symbol: 1},
        leg.vt_symbol,
    )
    engine: RecordingAlgoEngine = RecordingAlgoEngine()
    engine.contracts[leg.vt_symbol] = make_contract(leg.vt_symbol)
    algo: SpreadAlgoTemplate = make_algo(spread, engine, direction=direction, volume=2)

    assert algo.status == Status.NOTTRADED
    assert algo.traded == 0
    assert algo.traded_volume == 0
    assert algo.is_active()

    order: OrderData = attach_order(algo, leg.vt_symbol, 2, "1", trade_direction)
    algo.update_trade(make_trade(order, "1", 1, 10, trade_direction))

    assert algo.status == Status.PARTTRADED
    assert algo.traded == sign * 1
    assert algo.traded_volume == 1
    assert algo.traded_price == 10
    assert algo.is_active()
    assert algo.leg_orders[leg.vt_symbol] == [order.vt_orderid]

    algo.update_trade(make_trade(order, "2", 1, 12, trade_direction))

    assert algo.status == Status.ALLTRADED
    assert algo.traded == sign * 2
    assert algo.traded_volume == 2
    assert algo.traded_price == 11
    assert algo.leg_orders[leg.vt_symbol] == []
    assert not algo.is_active()

    algo.stop()
    assert algo.status == Status.ALLTRADED
    assert algo.stopped is False


def test_stop_cancels_active_orders_and_waits_for_terminal_status() -> None:
    leg: LegData = make_leg("leg.LOCAL")
    spread: SpreadData = make_spread(
        [leg],
        {leg.vt_symbol: 1},
        leg.vt_symbol,
    )
    engine: RecordingAlgoEngine = RecordingAlgoEngine()
    algo: SpreadAlgoTemplate = make_algo(spread, engine, volume=1)
    order: OrderData = attach_order(algo, leg.vt_symbol, 1, "working", Direction.LONG)

    algo.stop()

    assert algo.stopped is True
    assert algo.status == Status.NOTTRADED
    assert algo.is_active()
    assert engine.cancelled == [order.vt_orderid]
    assert algo.leg_orders[leg.vt_symbol] == [order.vt_orderid]

    order.status = Status.CANCELLED
    algo.update_order(order)

    assert algo.leg_orders[leg.vt_symbol] == []
    assert algo.status == Status.CANCELLED
    assert not algo.is_active()


def test_stop_stays_active_until_passive_hedge_catches_up() -> None:
    active: LegData = make_leg("active.LOCAL")
    passive: LegData = make_leg("passive.LOCAL")
    spread: SpreadData = make_spread(
        [active, passive],
        {
            active.vt_symbol: 1,
            passive.vt_symbol: 1,
        },
        active.vt_symbol,
    )
    engine: RecordingAlgoEngine = RecordingAlgoEngine()
    algo: SpreadAlgoTemplate = make_algo(spread, engine, volume=1)
    algo.leg_traded[active.vt_symbol] = 1

    algo.stop()

    assert algo.stopped is True
    assert algo.status == Status.NOTTRADED
    assert algo.is_active()
    assert engine.cancelled == []

    algo.send_order(active.vt_symbol, 9, 2, Direction.LONG)
    assert engine.sent == []
    assert algo.leg_orders[active.vt_symbol] == []

    algo.send_order(passive.vt_symbol, 10, 1, Direction.LONG)
    assert engine.sent == [(passive.vt_symbol, 10, 1, Direction.LONG)]
    assert algo.leg_orders[passive.vt_symbol] == engine.order_ids

    algo.leg_traded[passive.vt_symbol] = 1
    algo.leg_orders[passive.vt_symbol].clear()
    algo.check_algo_cancelled()

    assert algo.status == Status.CANCELLED
    assert not algo.is_active()


def test_rejected_order_leaves_active_list_without_changing_status() -> None:
    leg: LegData = make_leg("leg.LOCAL")
    spread: SpreadData = make_spread(
        [leg],
        {leg.vt_symbol: 1},
        leg.vt_symbol,
    )
    algo: SpreadAlgoTemplate = make_algo(spread, RecordingAlgoEngine(), volume=1)
    order: OrderData = attach_order(algo, leg.vt_symbol, 1, "reject", Direction.LONG)

    order.status = Status.REJECTED
    algo.update_order(order)

    assert algo.leg_orders[leg.vt_symbol] == []
    assert algo.status == Status.NOTTRADED
    assert algo.traded == 0
    assert algo.is_active()


def test_taker_interval_cancels_unfinished_leg_orders() -> None:
    leg: LegData = make_leg("leg.LOCAL")
    spread: SpreadData = make_spread(
        [leg],
        {leg.vt_symbol: 1},
        leg.vt_symbol,
    )
    engine: RecordingAlgoEngine = RecordingAlgoEngine()
    algo: SpreadTakerAlgo = make_algo(
        spread,
        engine,
        interval=1,
        algo_class=SpreadTakerAlgo,
    )
    order: OrderData = attach_order(algo, leg.vt_symbol, 1, "resting", Direction.LONG)

    algo.update_timer()
    assert algo.count == 1
    assert engine.cancelled == []

    algo.update_timer()
    assert algo.count == 0
    assert engine.cancelled == [order.vt_orderid]
    assert algo.leg_orders[leg.vt_symbol] == [order.vt_orderid]
    assert algo.status == Status.NOTTRADED
    assert algo.is_active()


def test_taker_sends_active_leg_only_when_ask_reaches_limit() -> None:
    leg: LegData = make_leg("active.LOCAL")
    spread: SpreadData = make_spread(
        [leg],
        {leg.vt_symbol: 1},
        leg.vt_symbol,
    )
    assert spread.calculate_price()
    engine: RecordingAlgoEngine = RecordingAlgoEngine()
    engine.contracts[leg.vt_symbol] = make_contract(leg.vt_symbol)
    tick: TickData = TickData(
        gateway_name="FAKE",
        symbol="active",
        exchange=Exchange.LOCAL,
        datetime=datetime(2024, 1, 2, 9, 0),
        bid_price_1=10,
        ask_price_1=11,
        bid_volume_1=12,
        ask_volume_1=9,
    )
    engine.ticks[leg.vt_symbol] = tick
    algo: SpreadTakerAlgo = make_algo(
        spread,
        engine,
        price=10,
        volume=4,
        payup=2,
        algo_class=SpreadTakerAlgo,
    )

    algo.on_tick(tick)
    assert engine.sent == []
    assert algo.leg_orders[leg.vt_symbol] == []
    assert algo.status == Status.NOTTRADED

    algo.price = spread.ask_price
    algo.on_tick(tick)

    assert engine.sent == [(leg.vt_symbol, 13, 4, Direction.LONG)]
    assert algo.leg_orders[leg.vt_symbol] == engine.order_ids
    assert algo.status == Status.NOTTRADED
    assert algo.traded == 0
    assert algo.is_active()
