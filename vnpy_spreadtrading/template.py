from abc import abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any
from collections.abc import Callable
from copy import copy

from vnpy.trader.object import (
    TickData, TradeData, OrderData, ContractData, BarData
)
from vnpy.trader.constant import Direction, Status, Interval
from vnpy.trader.utility import floor_to, ceil_to, round_to

from .base import SpreadData, LegData, EngineType, AlgoItem

if TYPE_CHECKING:
    from .engine import SpreadStrategyEngine


class SpreadAlgoTemplate:
    """
    Template for implementing spread trading algos.
    """
    algo_name: str = "AlgoTemplate"

    def __init__(
        self,
        algo_engine: Any,
        algoid: str,
        spread: SpreadData,
        direction: Direction,
        price: float,
        volume: float,
        payup: int,
        interval: int,
        lock: bool,
        extra: dict
    ) -> None:
        """"""
        self.algo_engine: Any = algo_engine
        self.algoid: str = algoid

        self.spread: SpreadData = spread
        self.spread_name: str = spread.name

        self.direction: Direction = direction
        self.price: float = price
        self.volume: float = volume
        self.payup: int = payup
        self.interval: int = interval
        self.lock: bool = lock

        if direction == Direction.LONG:
            self.target = volume
        else:
            self.target = -volume

        self.status: Status = Status.NOTTRADED  # 算法状态
        self.count: int = 0                     # 读秒计数
        self.traded: float = 0                  # 成交数量
        self.traded_volume: float = 0           # 成交数量（绝对值）
        self.traded_price: float = 0            # 成交价格
        self.stopped: bool = False              # 是否已被用户停止算法

        self.leg_traded: defaultdict = defaultdict(float)
        self.leg_cost: defaultdict = defaultdict(float)
        self.leg_orders: defaultdict = defaultdict(list)

        self.order_trade_volume: defaultdict = defaultdict(int)
        self.orders: dict[str, OrderData] = {}

        self.write_log("算法已启动")

    def get_item(self) -> AlgoItem:
        """获取数据对象"""
        item: AlgoItem = AlgoItem(
            algoid=self.algoid,
            spread_name=self.spread_name,
            direction=self.direction,
            price=self.price,
            payup=self.payup,
            volume=self.volume,
            traded_volume=self.traded_volume,
            traded_price=self.traded_price,
            interval=self.interval,
            count=self.count,
            status=self.status
        )
        return item

    def is_active(self) -> bool:
        """判断算法是否处于运行中"""
        if self.status not in [Status.CANCELLED, Status.ALLTRADED]:
            return True
        else:
            return False

    def is_order_finished(self) -> bool:
        """检查委托是否全部结束"""
        finished: bool = True

        for leg in self.spread.legs.values():
            vt_orderids: list = self.leg_orders[leg.vt_symbol]

            if vt_orderids:
                finished = False
                break

        return finished

    def is_hedge_finished(self) -> bool:
        """检查当前各条腿是否平衡"""
        active_symbol: str = self.spread.active_leg.vt_symbol
        active_traded: float = self.leg_traded[active_symbol]

        spread_volume: float = self.spread.calculate_spread_volume(
            active_symbol, active_traded
        )

        finished: bool = True

        for leg in self.spread.passive_legs:
            passive_symbol: str = leg.vt_symbol

            leg_target: float = self.spread.calculate_leg_volume(
                passive_symbol, spread_volume
            )
            leg_traded: float = self.leg_traded[passive_symbol]

            if leg_target > 0 and leg_traded < leg_target:
                finished = False
            elif leg_target < 0 and leg_traded > leg_target:
                finished = False

            if not finished:
                break

        return finished

    def check_algo_cancelled(self) -> None:
        """检查算法是否已停止"""
        if (
            self.stopped
            and self.is_order_finished()
            and self.is_hedge_finished()
        ):
            self.status = Status.CANCELLED
            self.write_log("算法已停止")
            self.put_event()

    def stop(self) -> None:
        """"""
        if not self.is_active():
            return

        self.write_log("算法停止中")
        self.stopped = True
        self.cancel_all_order()

        self.check_algo_cancelled()

    def update_tick(self, tick: TickData) -> None:
        """"""
        self.on_tick(tick)

    def update_trade(self, trade: TradeData) -> None:
        """"""
        trade_volume: float = trade.volume

        if trade.direction == Direction.LONG:
            self.leg_traded[trade.vt_symbol] += trade_volume
            self.leg_cost[trade.vt_symbol] += trade_volume * trade.price
        else:
            self.leg_traded[trade.vt_symbol] -= trade_volume
            self.leg_cost[trade.vt_symbol] -= trade_volume * trade.price

        self.calculate_traded_volume()
        self.calculate_traded_price()

        # Sum up total traded volume of each order,
        self.order_trade_volume[trade.vt_orderid] += trade.volume

        # Remove order from active list if all volume traded
        order: OrderData = self.orders[trade.vt_orderid]
        contract: ContractData | None = self.get_contract(trade.vt_symbol)

        if contract:
            trade_volume = round_to(
                self.order_trade_volume[order.vt_orderid],
                contract.min_volume
            )

        if trade_volume == order.volume:
            vt_orderids: list = self.leg_orders[order.vt_symbol]
            if order.vt_orderid in vt_orderids:
                vt_orderids.remove(order.vt_orderid)

        msg: str = f"委托成交[{trade.vt_orderid}]，{trade.vt_symbol}，{trade.direction.value}，{trade.volume}@{trade.price}"
        self.write_log(msg)

        self.put_event()
        self.on_trade(trade)

    def update_order(self, order: OrderData) -> None:
        """"""
        self.orders[order.vt_orderid] = order

        # Remove order from active list if rejected or cancelled
        if order.status in {Status.REJECTED, Status.CANCELLED}:
            vt_orderids: list = self.leg_orders[order.vt_symbol]
            if order.vt_orderid in vt_orderids:
                vt_orderids.remove(order.vt_orderid)

            msg: str = f"委托{order.status.value}[{order.vt_orderid}]"
            self.write_log(msg)

        self.on_order(order)

        # 如果在停止任务，则检查是否已经可以停止算法
        self.check_algo_cancelled()

    def update_timer(self) -> None:
        """"""
        self.count += 1
        if self.count > self.interval:
            self.count = 0
            self.on_interval()

        self.put_event()

    def put_event(self) -> None:
        """"""
        self.algo_engine.put_algo_event(self)

    def write_log(self, msg: str) -> None:
        """"""
        self.algo_engine.write_algo_log(self, msg)

    def send_order(
        self,
        vt_symbol: str,
        price: float,
        volume: float,
        direction: Direction,
        fak: bool = False
    ) -> None:
        """"""
        # 如果已经进入停止任务，禁止主动腿发单
        if self.stopped and vt_symbol == self.spread.active_leg.vt_symbol:
            return

        leg: LegData = self.spread.legs[vt_symbol]
        volume = round_to(volume, leg.min_volume)

        price = round_to(price, leg.pricetick)

        # 检查价格是否超过涨跌停板
        tick: TickData | None = self.get_tick(vt_symbol)

        if tick:
            if direction == Direction.LONG and tick.limit_up:
                price = min(price, tick.limit_up)
            elif direction == Direction.SHORT and tick.limit_down:
                price = max(price, tick.limit_down)

        vt_orderids: list = self.algo_engine.send_order(
            self,
            vt_symbol,
            price,
            volume,
            direction,
            self.lock,
            fak
        )

        self.leg_orders[vt_symbol].extend(vt_orderids)

        msg: str = "发出委托[{}]，{}，{}，{}@{}".format(
            "|".join(vt_orderids),
            vt_symbol,
            direction.value,
            volume,
            price
        )
        self.write_log(msg)

    def cancel_leg_order(self, vt_symbol: str) -> None:
        """"""
        for vt_orderid in self.leg_orders[vt_symbol]:
            self.algo_engine.cancel_order(self, vt_orderid)

    def cancel_all_order(self) -> None:
        """"""
        for vt_symbol in self.leg_orders.keys():
            self.cancel_leg_order(vt_symbol)

    def calculate_traded_volume(self) -> None:
        """"""
        self.traded = 0
        spread: SpreadData = self.spread

        n: int = 0
        for leg in spread.legs.values():
            leg_traded: float = self.leg_traded[leg.vt_symbol]
            trading_multiplier: int = spread.trading_multipliers[leg.vt_symbol]
            if not trading_multiplier:
                continue

            adjusted_leg_traded: float = leg_traded / trading_multiplier
            adjusted_leg_traded = round_to(adjusted_leg_traded, spread.min_volume)

            if adjusted_leg_traded > 0:
                adjusted_leg_traded = floor_to(adjusted_leg_traded, spread.min_volume)
            else:
                adjusted_leg_traded = ceil_to(adjusted_leg_traded, spread.min_volume)

            if not n:
                self.traded = adjusted_leg_traded
            else:
                if adjusted_leg_traded > 0:
                    self.traded = min(self.traded, adjusted_leg_traded)
                elif adjusted_leg_traded < 0:
                    self.traded = max(self.traded, adjusted_leg_traded)
                else:
                    self.traded = 0

            n += 1

        self.traded_volume = abs(self.traded)

        if self.target > 0 and self.traded >= self.target:
            self.status = Status.ALLTRADED
        elif self.target < 0 and self.traded <= self.target:
            self.status = Status.ALLTRADED
        elif not self.traded:
            self.status = Status.NOTTRADED
        else:
            self.status = Status.PARTTRADED

    def calculate_traded_price(self) -> None:
        """"""
        self.traded_price = 0
        spread: SpreadData = self.spread

        data: dict = {}

        for variable, vt_symbol in spread.variable_symbols.items():
            leg: LegData = spread.legs[vt_symbol]
            trading_multiplier: int = spread.trading_multipliers[leg.vt_symbol]

            # Use last price for non-trading leg (trading multiplier is 0)
            if not trading_multiplier:
                data[variable] = leg.tick.last_price
            else:
                # If any leg is not traded yet, clear data dict to set traded price to 0
                leg_traded: float = self.leg_traded[leg.vt_symbol]
                if not leg_traded:
                    data.clear()
                    break

                leg_cost: float = self.leg_cost[leg.vt_symbol]
                data[variable] = leg_cost / leg_traded

        if data:
            self.traded_price = spread.parse_formula(spread.price_code, data)
            self.traded_price = round_to(self.traded_price, spread.pricetick)
        else:
            self.traded_price = 0

    def get_tick(self, vt_symbol: str) -> TickData | None:
        """"""
        return self.algo_engine.get_tick(vt_symbol)

    def get_contract(self, vt_symbol: str) -> ContractData | None:
        """"""
        return self.algo_engine.get_contract(vt_symbol)

    def on_tick(self, tick: TickData) -> None:
        """"""
        return

    def on_order(self, order: OrderData) -> None:
        """"""
        return

    def on_trade(self, trade: TradeData) -> None:
        """"""
        return

    def on_interval(self) -> None:
        """"""
        return


class SpreadStrategyTemplate:
    """
    Template for implementing spread trading strategies.
    """

    author: str = ""
    parameters: list[str] = []
    variables: list[str] = []

    def __init__(
        self,
        strategy_engine: "SpreadStrategyEngine",
        strategy_name: str,
        spread: SpreadData,
        setting: dict
    ) -> None:
        """"""
        self.strategy_engine: SpreadStrategyEngine = strategy_engine
        self.strategy_name: str = strategy_name
        self.spread: SpreadData = spread
        self.spread_name: str = spread.name

        self.inited: bool = False
        self.trading: bool = False

        self.variables: list = copy(self.variables)
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")

        self.vt_orderids: set[str] = set()
        self.algoids: set[str] = set()

        self.update_setting(setting)

    def update_setting(self, setting: dict) -> None:
        """
        Update strategy parameter wtih value in setting dict.
        """
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

    @classmethod
    def get_class_parameters(cls) -> dict:
        """
        Get default parameters dict of strategy class.
        """
        class_parameters: dict = {}
        for name in cls.parameters:
            class_parameters[name] = getattr(cls, name)
        return class_parameters

    def get_parameters(self) -> dict:
        """
        Get strategy parameters dict.
        """
        strategy_parameters: dict = {}
        for name in self.parameters:
            strategy_parameters[name] = getattr(self, name)
        return strategy_parameters

    def get_variables(self) -> dict:
        """
        Get strategy variables dict.
        """
        strategy_variables: dict = {}
        for name in self.variables:
            strategy_variables[name] = getattr(self, name)
        return strategy_variables

    def get_data(self) -> dict:
        """
        Get strategy data.
        """
        strategy_data: dict = {
            "strategy_name": self.strategy_name,
            "spread_name": self.spread_name,
            "class_name": self.__class__.__name__,
            "author": self.author,
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
        }
        return strategy_data

    def update_spread_algo(self, algo: SpreadAlgoTemplate) -> None:
        """
        Callback when algo status is updated.
        """
        if not algo.is_active() and algo.algoid in self.algoids:
            self.algoids.remove(algo.algoid)

        self.on_spread_algo(algo)

    @abstractmethod
    def on_init(self) -> None:
        """
        Callback when strategy is inited.
        """
        return

    @abstractmethod
    def on_start(self) -> None:
        """
        Callback when strategy is started.
        """
        return

    @abstractmethod
    def on_stop(self) -> None:
        """
        Callback when strategy is stopped.
        """
        return

    @abstractmethod
    def on_spread_data(self) -> None:
        """
        Callback when spread price is updated.
        """
        return

    @abstractmethod
    def on_spread_tick(self, tick: TickData) -> None:
        """
        Callback when new spread tick data is generated.
        """
        return

    @abstractmethod
    def on_spread_bar(self, bar: BarData) -> None:
        """
        Callback when new spread bar data is generated.
        """
        return

    @abstractmethod
    def on_spread_pos(self) -> None:
        """
        Callback when spread position is updated.
        """
        return

    @abstractmethod
    def on_spread_algo(self, algo: SpreadAlgoTemplate) -> None:
        """
        Callback when algo status is updated.
        """
        return

    def start_algo(
        self,
        direction: Direction,
        price: float,
        volume: float,
        payup: int,
        interval: int,
        lock: bool,
        extra: dict
    ) -> str:
        """"""
        if not self.trading:
            return ""

        algoid: str = self.strategy_engine.start_algo(
            self,
            self.spread_name,
            direction,
            price,
            volume,
            payup,
            interval,
            lock,
            extra
        )

        self.algoids.add(algoid)

        return algoid

    def start_long_algo(
        self,
        price: float,
        volume: float,
        payup: int,
        interval: int,
        lock: bool = False,
        extra: dict | None = None
    ) -> str:
        """"""
        if not extra:
            extra = {}

        return self.start_algo(
            Direction.LONG, price, volume,
            payup, interval, lock, extra
        )

    def start_short_algo(
        self,
        price: float,
        volume: float,
        payup: int,
        interval: int,
        lock: bool = False,
        extra: dict | None = None
    ) -> str:
        """"""
        if not extra:
            extra = {}

        return self.start_algo(
            Direction.SHORT, price, volume,
            payup, interval, lock, extra
        )

    def stop_algo(self, algoid: str) -> None:
        """"""
        if not self.trading:
            return

        self.strategy_engine.stop_algo(self, algoid)

    def stop_all_algos(self) -> None:
        """"""
        for algoid in list(self.algoids):
            self.stop_algo(algoid)

    def put_event(self) -> None:
        """"""
        self.strategy_engine.put_strategy_event(self)

    def write_log(self, msg: str) -> None:
        """"""
        self.strategy_engine.write_strategy_log(self, msg)

    def get_engine_type(self) -> EngineType:
        """"""
        return self.strategy_engine.get_engine_type()

    def get_spread_tick(self) -> TickData:
        """"""
        return self.spread.to_tick()

    def get_spread_pos(self) -> float:
        """"""
        return self.spread.net_pos

    def get_leg_tick(self, vt_symbol: str) -> TickData | None:
        """"""
        leg: LegData | None = self.spread.legs.get(vt_symbol, None)

        if not leg:
            return None

        return leg.tick

    def get_leg_pos(self, vt_symbol: str, direction: Direction = Direction.NET) -> float | None:
        """"""
        leg: LegData | None = self.spread.legs.get(vt_symbol, None)

        if not leg:
            return None

        if direction == Direction.NET:
            return leg.net_pos
        elif direction == Direction.LONG:
            return leg.long_pos
        else:
            return leg.short_pos

    def send_email(self, msg: str) -> None:
        """
        Send email to default receiver.
        """
        if self.inited:
            self.strategy_engine.send_email(msg, self)

    def load_bar(
        self,
        days: int,
        interval: Interval = Interval.MINUTE,
        callback: Callable | None = None,
    ) -> None:
        """
        Load historical bar data for initializing strategy.
        """
        if callback is None:
            callback = self.on_spread_bar

        self.strategy_engine.load_bar(self.spread, days, interval, callback)

    def load_tick(self, days: int) -> None:
        """
        Load historical tick data for initializing strategy.
        """
        self.strategy_engine.load_tick(self.spread, days, self.on_spread_tick)
