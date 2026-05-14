import pytest

from vnpy.trader.constant import Direction

from vnpy_spreadtrading.algo import SpreadTakerAlgo
from vnpy_spreadtrading.base import LegData, SpreadData
from vnpy_spreadtrading.template import SpreadAlgoTemplate


class FakeAlgoEngine:
    def write_algo_log(self, algo: SpreadAlgoTemplate, msg: str) -> None:
        pass

    def put_algo_event(self, algo: SpreadAlgoTemplate) -> None:
        pass

    def get_tick(self, vt_symbol: str) -> None:
        return None

    def get_contract(self, vt_symbol: str) -> None:
        return None

    def send_order(self, *args: object, **kwargs: object) -> list[str]:
        return []

    def cancel_order(self, algo: SpreadAlgoTemplate, vt_orderid: str) -> None:
        pass


class RecordingTakerAlgo(SpreadTakerAlgo):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.sent_leg_orders: list[tuple[str, float]] = []
        super().__init__(*args, **kwargs)

    def send_leg_order(self, vt_symbol: str, leg_volume: float) -> None:
        self.sent_leg_orders.append((vt_symbol, leg_volume))


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

    price_formula: str = "+".join(formula_parts)

    return SpreadData(
        "spread",
        legs,
        variable_symbols,
        variable_directions,
        price_formula,
        trading_multipliers,
        active_symbol,
        min_volume,
    )


def make_algo(spread: SpreadData, volume: float = 1) -> SpreadAlgoTemplate:
    return SpreadAlgoTemplate(
        FakeAlgoEngine(),
        "algo",
        spread,
        Direction.LONG,
        0,
        volume,
        0,
        1,
        False,
        {},
    )


def make_taker_algo(spread: SpreadData, volume: float = 1) -> RecordingTakerAlgo:
    return RecordingTakerAlgo(
        FakeAlgoEngine(),
        "algo",
        spread,
        Direction.LONG,
        0,
        volume,
        0,
        1,
        False,
        {},
    )


@pytest.mark.parametrize(
    ("passive_traded", "expected_traded"),
    [
        (2, 0),
        (3, 1),
    ],
)
def test_calculate_traded_volume_requires_full_multiplier(
    passive_traded: float,
    expected_traded: float,
) -> None:
    active_leg: LegData = make_leg("active.LOCAL")
    passive_leg: LegData = make_leg("passive.LOCAL")
    spread: SpreadData = make_spread(
        [active_leg, passive_leg],
        {
            active_leg.vt_symbol: 1,
            passive_leg.vt_symbol: 3,
        },
        active_leg.vt_symbol,
    )
    algo: SpreadAlgoTemplate = make_algo(spread)

    algo.leg_traded[active_leg.vt_symbol] = 1
    algo.leg_traded[passive_leg.vt_symbol] = passive_traded
    algo.calculate_traded_volume()

    assert algo.traded == expected_traded


@pytest.mark.parametrize(
    ("passive_traded", "expected_traded"),
    [
        (-2, 0),
        (-3, 1),
    ],
)
def test_calculate_traded_volume_preserves_negative_multiplier_direction(
    passive_traded: float,
    expected_traded: float,
) -> None:
    active_leg: LegData = make_leg("active.LOCAL")
    passive_leg: LegData = make_leg("passive.LOCAL")
    spread: SpreadData = make_spread(
        [active_leg, passive_leg],
        {
            active_leg.vt_symbol: 1,
            passive_leg.vt_symbol: -3,
        },
        active_leg.vt_symbol,
    )
    algo: SpreadAlgoTemplate = make_algo(spread)

    algo.leg_traded[active_leg.vt_symbol] = 1
    algo.leg_traded[passive_leg.vt_symbol] = passive_traded
    algo.calculate_traded_volume()

    assert algo.traded == expected_traded


@pytest.mark.parametrize(
    ("passive_traded", "expected_traded"),
    [
        (0.2, 0),
        (0.3, 0.1),
    ],
)
def test_calculate_traded_volume_supports_decimal_min_volume(
    passive_traded: float,
    expected_traded: float,
) -> None:
    active_leg: LegData = make_leg("active.LOCAL", min_volume=0.1)
    passive_leg: LegData = make_leg("passive.LOCAL", min_volume=0.1)
    spread: SpreadData = make_spread(
        [active_leg, passive_leg],
        {
            active_leg.vt_symbol: 1,
            passive_leg.vt_symbol: 3,
        },
        active_leg.vt_symbol,
        min_volume=0.1,
    )
    algo: SpreadAlgoTemplate = make_algo(spread, volume=0.1)

    algo.leg_traded[active_leg.vt_symbol] = 0.1
    algo.leg_traded[passive_leg.vt_symbol] = passive_traded
    algo.calculate_traded_volume()

    assert algo.traded == pytest.approx(expected_traded)


def test_calculate_pos_ignores_non_trading_leg_when_initializing() -> None:
    non_trading_leg: LegData = make_leg("non_trading.LOCAL")
    trading_leg: LegData = make_leg("trading.LOCAL")
    spread: SpreadData = make_spread(
        [non_trading_leg, trading_leg],
        {
            non_trading_leg.vt_symbol: 0,
            trading_leg.vt_symbol: 1,
        },
        trading_leg.vt_symbol,
    )

    spread.leg_pos[trading_leg.vt_symbol] = 2
    spread.calculate_pos()

    assert spread.long_pos == 2
    assert spread.net_pos == 2


def test_calculate_pos_preserves_negative_multiplier_direction() -> None:
    leg: LegData = make_leg("leg.LOCAL")
    spread: SpreadData = make_spread(
        [leg],
        {leg.vt_symbol: -3},
        leg.vt_symbol,
    )

    spread.leg_pos[leg.vt_symbol] = -3
    spread.calculate_pos()

    assert spread.long_pos == 1
    assert spread.net_pos == 1


def test_calculate_price_uses_absolute_multiplier_for_capacity() -> None:
    leg: LegData = make_leg("leg.LOCAL")
    spread: SpreadData = make_spread(
        [leg],
        {leg.vt_symbol: -3},
        leg.vt_symbol,
    )

    assert spread.calculate_price()
    assert spread.bid_volume == 3
    assert spread.ask_volume == 4


def test_hedge_passive_legs_does_not_round_partial_active_leg_to_spread_volume() -> None:
    active_leg: LegData = make_leg("active.LOCAL", min_volume=0.1)
    passive_leg: LegData = make_leg("passive.LOCAL", min_volume=0.1)
    spread: SpreadData = make_spread(
        [active_leg, passive_leg],
        {
            active_leg.vt_symbol: 1,
            passive_leg.vt_symbol: 1,
        },
        active_leg.vt_symbol,
        min_volume=1,
    )
    algo: RecordingTakerAlgo = make_taker_algo(spread)

    algo.leg_traded[active_leg.vt_symbol] = 0.6
    algo.hedge_passive_legs()
    assert algo.sent_leg_orders == []

    algo.leg_traded[active_leg.vt_symbol] = 1.0
    algo.hedge_passive_legs()
    assert algo.sent_leg_orders == [(passive_leg.vt_symbol, 1)]
