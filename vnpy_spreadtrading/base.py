from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
from tzlocal import get_localzone_name
from dataclasses import dataclass

from vnpy.trader.object import (
    HistoryRequest, TickData, PositionData, TradeData, ContractData, BarData
)
from vnpy.trader.constant import Direction, Offset, Exchange, Interval, Status
from vnpy.trader.utility import floor_to, ceil_to, round_to, extract_vt_symbol, ZoneInfo
from vnpy.trader.database import BaseDatabase, get_database
from vnpy.trader.datafeed import BaseDatafeed, get_datafeed


EVENT_SPREAD_DATA = "eSpreadData"
EVENT_SPREAD_POS = "eSpreadPos"
EVENT_SPREAD_LOG = "eSpreadLog"
EVENT_SPREAD_ALGO = "eSpreadAlgo"
EVENT_SPREAD_STRATEGY = "eSpreadStrategy"

LOCAL_TZ = ZoneInfo(get_localzone_name())


class LegData:
    """"""

    def __init__(self, vt_symbol: str) -> None:
        """"""
        self.vt_symbol: str = vt_symbol

        # Price and position data
        self.bid_price: float = 0
        self.ask_price: float = 0
        self.bid_volume: float = 0
        self.ask_volume: float = 0

        self.long_pos: float = 0
        self.short_pos: float = 0
        self.net_pos: float = 0

        self.last_price: float = 0
        self.net_pos_price: float = 0       # Average entry price of net position

        # Tick data buf
        self.tick: TickData = None

        # Contract data
        self.size: float = 0
        self.net_position: bool = False
        self.min_volume: float = 0
        self.pricetick: float = 0

    def update_contract(self, contract: ContractData) -> None:
        """"""
        self.size = contract.size
        self.net_position = contract.net_position
        self.min_volume = contract.min_volume
        self.pricetick = contract.pricetick

    def update_tick(self, tick: TickData) -> None:
        """"""
        self.bid_price = tick.bid_price_1
        self.ask_price = tick.ask_price_1
        self.bid_volume = tick.bid_volume_1
        self.ask_volume = tick.ask_volume_1
        self.last_price = tick.last_price

        self.tick = tick

    def update_position(self, position: PositionData) -> None:
        """"""
        if position.direction == Direction.NET:
            self.net_pos = position.volume
            self.net_pos_price = position.price
        else:
            if position.direction == Direction.LONG:
                self.long_pos = position.volume
            else:
                self.short_pos = position.volume
            self.net_pos = self.long_pos - self.short_pos

    def update_trade(self, trade: TradeData) -> None:
        """"""
        # Only update net pos for contract with net position mode
        if self.net_position:
            trade_cost: float = trade.volume * trade.price
            old_cost: float = self.net_pos * self.net_pos_price

            if trade.direction == Direction.LONG:
                new_pos: float = self.net_pos + trade.volume

                if self.net_pos >= 0:
                    new_cost = old_cost + trade_cost
                    self.net_pos_price = new_cost / new_pos
                else:
                    # If all previous short position closed
                    if not new_pos:
                        self.net_pos_price = 0
                    # If only part short position closed
                    elif new_pos > 0:
                        self.net_pos_price = trade.price
            else:
                new_pos: float = self.net_pos - trade.volume

                if self.net_pos <= 0:
                    new_cost = old_cost - trade_cost
                    self.net_pos_price = new_cost / new_pos
                else:
                    # If all previous long position closed
                    if not new_pos:
                        self.net_pos_price = 0
                    # If only part long position closed
                    elif new_pos < 0:
                        self.net_pos_price = trade.price

            self.net_pos = new_pos
        else:
            if trade.direction == Direction.LONG:
                if trade.offset == Offset.OPEN:
                    self.long_pos += trade.volume
                else:
                    self.short_pos -= trade.volume
            else:
                if trade.offset == Offset.OPEN:
                    self.short_pos += trade.volume
                else:
                    self.long_pos -= trade.volume

            self.net_pos = self.long_pos - self.short_pos


class SpreadData:
    """"""

    def __init__(
        self,
        name: str,
        legs: List[LegData],
        variable_symbols: Dict[str, str],
        variable_directions: Dict[str, int],
        price_formula: str,
        trading_multipliers: Dict[str, int],
        active_symbol: str,
        min_volume: float,
        compile_formula: bool = True
    ) -> None:
        """"""
        self.name: str = name
        self.compile_formula: bool = compile_formula

        self.legs: Dict[str, LegData] = {}
        self.active_leg: LegData = None
        self.passive_legs: List[LegData] = []

        self.min_volume: float = min_volume
        self.pricetick: float = 0

        # For calculating spread pos and sending orders
        self.trading_multipliers: Dict[str, int] = trading_multipliers

        self.price_formula: str = ""
        self.trading_formula: str = ""

        for leg in legs:
            self.legs[leg.vt_symbol] = leg
            if leg.vt_symbol == active_symbol:
                self.active_leg = leg
            else:
                self.passive_legs.append(leg)

            trading_multiplier: int = self.trading_multipliers[leg.vt_symbol]
            if trading_multiplier > 0:
                self.trading_formula += f"+{trading_multiplier}*{leg.vt_symbol}"
            else:
                self.trading_formula += f"{trading_multiplier}*{leg.vt_symbol}"

            if not self.pricetick:
                self.pricetick = leg.pricetick
            else:
                self.pricetick = min(self.pricetick, leg.pricetick)

        # Spread data
        self.bid_price: float = 0
        self.ask_price: float = 0
        self.bid_volume: float = 0
        self.ask_volume: float = 0

        self.long_pos: int = 0
        self.short_pos: int = 0
        self.net_pos: int = 0

        self.datetime: datetime = None

        self.leg_pos: defaultdict = defaultdict(int)

        # 价差计算公式相关
        self.variable_symbols: dict = variable_symbols
        self.variable_directions: dict = variable_directions
        self.price_formula = price_formula

        # 实盘时编译公式，加速计算
        if compile_formula:
            self.price_code: str = compile(price_formula, __name__, "eval")
        # 回测时不编译公式，从而支持多进程优化
        else:
            self.price_code: str = price_formula

        self.variable_legs: Dict[str, LegData] = {}
        for variable, vt_symbol in variable_symbols.items():
            leg: LegData = self.legs[vt_symbol]
            self.variable_legs[variable] = leg

    def calculate_price(self) -> bool:
        """
        计算价差盘口

        1. 如果各条腿价格均有效，则计算成功，返回True
        2. 反之只要有一条腿的价格无效，则计算失败，返回False
        """
        self.clear_price()

        # Go through all legs to calculate price
        bid_data: dict = {}
        ask_data: dict = {}
        volume_inited: bool = False

        for variable, leg in self.variable_legs.items():
            # Filter not all leg price data has been received
            if not leg.bid_volume or not leg.ask_volume:
                self.clear_price()
                return False

            # Generate price dict for calculating spread bid/ask
            variable_direction: int = self.variable_directions[variable]
            if variable_direction > 0:
                bid_data[variable] = leg.bid_price
                ask_data[variable] = leg.ask_price
            else:
                bid_data[variable] = leg.ask_price
                ask_data[variable] = leg.bid_price

            # Calculate volume
            trading_multiplier: int = self.trading_multipliers[leg.vt_symbol]
            if not trading_multiplier:
                continue

            leg_bid_volume: float = leg.bid_volume
            leg_ask_volume: float = leg.ask_volume

            if trading_multiplier > 0:
                adjusted_bid_volume: float = floor_to(
                    leg_bid_volume / trading_multiplier,
                    self.min_volume
                )
                adjusted_ask_volume: float = floor_to(
                    leg_ask_volume / trading_multiplier,
                    self.min_volume
                )
            else:
                adjusted_bid_volume: float = floor_to(
                    leg_ask_volume / abs(trading_multiplier),
                    self.min_volume
                )
                adjusted_ask_volume: float = floor_to(
                    leg_bid_volume / abs(trading_multiplier),
                    self.min_volume
                )

            # For the first leg, just initialize
            if not volume_inited:
                self.bid_volume = adjusted_bid_volume
                self.ask_volume = adjusted_ask_volume
                volume_inited = True
            # For following legs, use min value of each leg quoting volume
            else:
                self.bid_volume = min(self.bid_volume, adjusted_bid_volume)
                self.ask_volume = min(self.ask_volume, adjusted_ask_volume)

        # Calculate spread price
        self.bid_price = self.parse_formula(self.price_code, bid_data)
        self.ask_price = self.parse_formula(self.price_code, ask_data)

        # Round price to pricetick
        if self.pricetick:
            self.bid_price = round_to(self.bid_price, self.pricetick)
            self.ask_price = round_to(self.ask_price, self.pricetick)

        # Update calculate time
        self.datetime = datetime.now(LOCAL_TZ)

        return True

    def update_trade(self, trade: TradeData) -> None:
        """更新委托成交"""
        if trade.direction == Direction.LONG:
            self.leg_pos[trade.vt_symbol] += trade.volume
        else:
            self.leg_pos[trade.vt_symbol] -= trade.volume

    def calculate_pos(self) -> None:
        """"""
        long_pos = 0
        short_pos = 0

        for n, leg in enumerate(self.legs.values()):
            leg_long_pos = 0
            leg_short_pos = 0

            trading_multiplier: int = self.trading_multipliers[leg.vt_symbol]
            if not trading_multiplier:
                continue

            net_pos = self.leg_pos[leg.vt_symbol]
            adjusted_net_pos = net_pos / trading_multiplier

            if adjusted_net_pos > 0:
                adjusted_net_pos = floor_to(adjusted_net_pos, self.min_volume)
                leg_long_pos = adjusted_net_pos
            else:
                adjusted_net_pos = ceil_to(adjusted_net_pos, self.min_volume)
                leg_short_pos = abs(adjusted_net_pos)

            if not n:
                long_pos = leg_long_pos
                short_pos = leg_short_pos
            else:
                long_pos = min(long_pos, leg_long_pos)
                short_pos = min(short_pos, leg_short_pos)

        self.long_pos = long_pos
        self.short_pos = short_pos
        self.net_pos = long_pos - short_pos

    def clear_price(self) -> None:
        """"""
        self.bid_price = 0
        self.ask_price = 0
        self.bid_volume = 0
        self.ask_volume = 0

    def calculate_leg_volume(self, vt_symbol: str, spread_volume: float) -> float:
        """"""
        leg: LegData = self.legs[vt_symbol]
        trading_multiplier: int = self.trading_multipliers[leg.vt_symbol]
        leg_volume: float = spread_volume * trading_multiplier
        return leg_volume

    def calculate_spread_volume(self, vt_symbol: str, leg_volume: float) -> float:
        """"""
        leg: LegData = self.legs[vt_symbol]
        trading_multiplier: int = self.trading_multipliers[leg.vt_symbol]
        spread_volume: float = leg_volume / trading_multiplier

        if spread_volume > 0:
            spread_volume = floor_to(spread_volume, self.min_volume)
        else:
            spread_volume = ceil_to(spread_volume, self.min_volume)

        return spread_volume

    def to_tick(self) -> None:
        """"""
        tick: TickData = TickData(
            symbol=self.name,
            exchange=Exchange.LOCAL,
            datetime=self.datetime,
            name=self.name,
            last_price=(self.bid_price + self.ask_price) / 2,
            bid_price_1=self.bid_price,
            ask_price_1=self.ask_price,
            bid_volume_1=self.bid_volume,
            ask_volume_1=self.ask_volume,
            gateway_name="SPREAD"
        )
        return tick

    def get_leg_size(self, vt_symbol: str) -> float:
        """"""
        leg: LegData = self.legs[vt_symbol]
        return leg.size

    def parse_formula(self, formula: str, data: Dict[str, float]) -> Any:
        """"""
        locals().update(data)
        value = eval(formula)
        return value

    def get_item(self) -> None:
        """获取数据对象"""
        item: SpreadItem = SpreadItem(
            name=self.name,
            bid_volume=self.bid_volume,
            bid_price=self.bid_price,
            ask_price=self.ask_price,
            ask_volume=self.ask_volume,
            net_pos=self.net_pos,
            datetime=self.datetime,
            price_formula=self.price_formula,
            trading_formula=self.trading_formula,
        )
        return item


class EngineType(Enum):
    LIVE = "实盘"
    BACKTESTING = "回测"


class BacktestingMode(Enum):
    BAR = 1
    TICK = 2


def load_bar_data(
    spread: SpreadData,
    interval: Interval,
    start: datetime,
    end: datetime,
    pricetick: float = 0,
    output: Callable = print,
    backtesting: bool = False
) -> List[BarData]:
    """"""
    database: BaseDatabase = get_database()

    # Load bar data of each spread leg
    leg_bars: Dict[str, Dict] = {}

    for vt_symbol in spread.legs.keys():
        symbol, exchange = extract_vt_symbol(vt_symbol)

        # 初始化K线列表
        bar_data: List[BarData] = []

        # 只有实盘才优先尝试从数据服务查询
        if not backtesting:
            bar_data = query_bar_from_datafeed(
                symbol, exchange, interval, start, end, output
            )

        # 如果查询失败，则尝试从数据库中读取
        if not bar_data:
            bar_data = database.load_bar_data(
                symbol, exchange, interval, start, end
            )

        bars: Dict[datetime, BarData] = {bar.datetime: bar for bar in bar_data}
        leg_bars[vt_symbol] = bars

    # Calculate spread bar data
    spread_bars: List[BarData] = []

    for dt in bars.keys():
        spread_price = 0
        spread_value = 0
        spread_available: bool = True

        leg_data: dict = {}
        for variable, leg in spread.variable_legs.items():
            leg_bar: Optional[BarData] = leg_bars[leg.vt_symbol].get(dt, None)

            if leg_bar:
                # 缓存该腿当前的价格
                leg_data[variable] = leg_bar.close_price

                # 基于交易乘数累计价值
                trading_multiplier: int = spread.trading_multipliers[leg.vt_symbol]
                spread_value += trading_multiplier * leg_bar.close_price
            else:
                spread_available = False

        if spread_available:
            spread_price = spread.parse_formula(spread.price_code, leg_data)
            if pricetick:
                spread_price: float = round_to(spread_price, pricetick)

            spread_bar: BarData = BarData(
                symbol=spread.name,
                exchange=exchange.LOCAL,
                datetime=dt,
                interval=interval,
                open_price=spread_price,
                high_price=spread_price,
                low_price=spread_price,
                close_price=spread_price,
                gateway_name="SPREAD",
            )
            spread_bar.value = spread_value
            spread_bars.append(spread_bar)

    return spread_bars


def load_tick_data(
    spread: SpreadData,
    start: datetime,
    end: datetime
) -> List[TickData]:
    """"""
    database: BaseDatabase = get_database()
    return database.load_tick_data(
        spread.name, Exchange.LOCAL, start, end
    )


def query_bar_from_datafeed(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime,
    output: Callable = print
) -> List[BarData]:
    """
    Query bar data from RQData.
    """
    datafeed: BaseDatafeed = get_datafeed()

    req: HistoryRequest = HistoryRequest(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start=start,
        end=end
    )
    data: List[BarData] = datafeed.query_bar_history(req, output)
    return data


@dataclass
class SpreadItem:
    """价差数据容器"""

    name: str
    bid_volume: int
    bid_price: float
    ask_price: float
    ask_volume: int
    net_pos: int
    datetime: datetime
    price_formula: str
    trading_formula: str


@dataclass
class AlgoItem:
    """算法数据容器"""

    algoid: str
    spread_name: str
    direction: Direction
    price: float
    payup: int
    volume: float
    traded_volume: float
    traded_price: float
    interval: int
    count: int
    status: Status
