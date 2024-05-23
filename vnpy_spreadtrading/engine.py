import traceback
import importlib
import os
from types import ModuleType
from typing import List, Dict, Set, Callable, Any, Optional
from collections import defaultdict
from copy import copy
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future

from vnpy.event import EventEngine, Event
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.event import (
    EVENT_TICK, EVENT_POSITION, EVENT_CONTRACT,
    EVENT_ORDER, EVENT_TRADE, EVENT_TIMER
)
from vnpy.trader.utility import load_json, save_json
from vnpy.trader.object import (
    TickData, ContractData, BarData,
    PositionData, OrderData, TradeData, LogData,
    SubscribeRequest, OrderRequest, CancelRequest
)
from vnpy.trader.constant import (
    Direction, Offset, OrderType, Interval
)
from vnpy.trader.database import DB_TZ

from .base import (
    LegData, SpreadData,
    EVENT_SPREAD_DATA, EVENT_SPREAD_POS,
    EVENT_SPREAD_ALGO, EVENT_SPREAD_LOG,
    EVENT_SPREAD_STRATEGY,
    load_bar_data, load_tick_data,
    EngineType
)
from .template import SpreadAlgoTemplate, SpreadStrategyTemplate
from .algo import SpreadTakerAlgo


APP_NAME = "SpreadTrading"


class SpreadEngine(BaseEngine):
    """"""

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """Constructor"""
        super().__init__(main_engine, event_engine, APP_NAME)

        self.active: bool = False

        self.init_data_engine()
        self.init_algo_engine()
        self.init_strategy_engine()

    def init_data_engine(self) -> None:
        """初始化数据引擎"""
        self.data_engine: SpreadDataEngine = SpreadDataEngine(self)

        self.add_spread = self.data_engine.add_spread
        self.remove_spread = self.data_engine.remove_spread
        self.get_spread = self.data_engine.get_spread
        self.get_all_spread_names = self.data_engine.get_all_spread_names

    def init_algo_engine(self) -> None:
        """初始化算法引擎"""
        self.algo_engine: SpreadAlgoEngine = SpreadAlgoEngine(self)

        self.start_algo = self.algo_engine.start_algo
        self.stop_algo = self.algo_engine.stop_algo

    def init_strategy_engine(self) -> None:
        """初始化策略引擎"""
        self.strategy_engine: SpreadStrategyEngine = SpreadStrategyEngine(self)

        self.get_all_strategy_class_names = self.strategy_engine.get_all_strategy_class_names
        self.get_strategy_class_parameters = self.strategy_engine.get_strategy_class_parameters
        self.init_all_strategies = self.strategy_engine.init_all_strategies
        self.start_all_strategies = self.strategy_engine.start_all_strategies
        self.stop_all_strategies = self.strategy_engine.stop_all_strategies
        self.add_strategy = self.strategy_engine.add_strategy
        self.init_strategy = self.strategy_engine.init_strategy
        self.start_strategy = self.strategy_engine.start_strategy
        self.stop_strategy = self.strategy_engine.stop_strategy
        self.get_strategy_parameters = self.strategy_engine.get_strategy_parameters
        self.edit_strategy = self.strategy_engine.edit_strategy
        self.remove_strategy = self.strategy_engine.remove_strategy

    def start(self) -> None:
        """"""
        if self.active:
            return
        self.active = True

        self.data_engine.start()
        self.algo_engine.start()
        self.strategy_engine.start()

    def stop(self) -> None:
        """"""
        self.data_engine.stop()
        self.algo_engine.stop()
        self.strategy_engine.stop()

    def write_log(self, msg: str) -> None:
        """"""
        log: LegData = LogData(
            msg=msg,
            gateway_name=APP_NAME
        )
        event: Event = Event(EVENT_SPREAD_LOG, log)
        self.event_engine.put(event)

    def update_spread_data(self, spread: SpreadData) -> None:
        """"""
        self.algo_engine.update_spread_data(spread)
        self.strategy_engine.update_spread_data(spread)

    def update_spread_pos(self, spread: SpreadData) -> None:
        """"""
        self.strategy_engine.update_spread_pos(spread)

    def update_spread_algo(self, algo: SpreadAlgoTemplate) -> None:
        """"""
        self.strategy_engine.update_spread_algo(algo)


class SpreadDataEngine:
    """"""
    setting_filename: str = "spread_trading_setting.json"
    pos_filename: str = "spread_trading_pos.json"

    def __init__(self, spread_engine: SpreadEngine) -> None:
        """"""
        self.spread_engine: SpreadEngine = spread_engine
        self.main_engine: MainEngine = spread_engine.main_engine
        self.event_engine: EventEngine = spread_engine.event_engine

        self.write_log = spread_engine.write_log

        self.legs: Dict[str, LegData] = {}          # vt_symbol: leg
        self.spreads: Dict[str, SpreadData] = {}    # name: spread
        self.symbol_spread_map: Dict[str, List[SpreadData]] = defaultdict(list)
        self.order_spread_map: Dict[str, SpreadData] = {}

        self.tradeid_history: Set[str] = set()

    def start(self) -> None:
        """"""
        self.load_setting()
        self.load_pos()
        self.register_event()

        self.write_log("价差数据引擎启动成功")

    def stop(self) -> None:
        """"""
        pass

    def load_setting(self) -> None:
        """"""
        setting: dict = load_json(self.setting_filename)

        for spread_setting in setting:
            self.add_spread(
                spread_setting["name"],
                spread_setting["leg_settings"],
                spread_setting["price_formula"],
                spread_setting["active_symbol"],
                spread_setting.get("min_volume", 1),
                save=False
            )

    def save_setting(self) -> None:
        """"""
        setting: list = []

        for spread in self.spreads.values():
            leg_settings: list = []
            for variable, vt_symbol in spread.variable_symbols.items():
                trading_direction: int = spread.variable_directions[variable]
                trading_multiplier: int = spread.trading_multipliers[vt_symbol]

                leg_setting: dict = {
                    "variable": variable,
                    "vt_symbol": vt_symbol,
                    "trading_direction": trading_direction,
                    "trading_multiplier": trading_multiplier
                }
                leg_settings.append(leg_setting)

            spread_setting: dict = {
                "name": spread.name,
                "leg_settings": leg_settings,
                "price_formula": spread.price_formula,
                "active_symbol": spread.active_leg.vt_symbol,
                "min_volume": spread.min_volume
            }

            setting.append(spread_setting)

        save_json(self.setting_filename, setting)

    def save_pos(self) -> None:
        """保存价差持仓"""
        pos_data: dict = {}

        for spread in self.spreads.values():
            pos_data[spread.name] = spread.leg_pos

        save_json(self.pos_filename, pos_data)

    def load_pos(self) -> None:
        """加载价差持仓"""
        pos_data: dict = load_json(self.pos_filename)

        for name, leg_pos in pos_data.items():
            spread: SpreadData = self.spreads.get(name, None)
            if spread:
                spread.leg_pos.update(leg_pos)

    def register_event(self) -> None:
        """"""
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_POSITION, self.process_position_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)

    def process_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data

        leg: LegData = self.legs.get(tick.vt_symbol, None)
        if not leg:
            return
        leg.update_tick(tick)

        for spread in self.symbol_spread_map[tick.vt_symbol]:
            # 只有能成功计算出价差盘口时，才会送事件
            if spread.calculate_price():
                self.put_data_event(spread)

    def process_position_event(self, event: Event) -> None:
        """"""
        position: PositionData = event.data

        leg: LegData = self.legs.get(position.vt_symbol, None)
        if not leg:
            return
        leg.update_position(position)

        for spread in self.symbol_spread_map[position.vt_symbol]:
            spread.calculate_pos()
            self.put_pos_event(spread)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data

        if trade.vt_tradeid in self.tradeid_history:
            return
        self.tradeid_history.add(trade.vt_tradeid)

        # 查询该笔成交，对应的价差，并更新计算价差持仓
        spread: SpreadData = self.order_spread_map.get(trade.vt_orderid, None)
        if spread:
            spread.update_trade(trade)
            spread.calculate_pos()
            self.put_pos_event(spread)

            self.save_pos()

    def process_contract_event(self, event: Event) -> None:
        """"""
        contract: ContractData = event.data
        leg: LegData = self.legs.get(contract.vt_symbol, None)

        if leg:
            # Update contract data
            leg.update_contract(contract)

            req: SubscribeRequest = SubscribeRequest(
                contract.symbol, contract.exchange
            )
            self.main_engine.subscribe(req, contract.gateway_name)

    def put_data_event(self, spread: SpreadData) -> None:
        """"""
        self.spread_engine.update_spread_data(spread)

        event: Event = Event(EVENT_SPREAD_DATA, spread.get_item())
        self.event_engine.put(event)

    def put_pos_event(self, spread: SpreadData) -> None:
        """"""
        self.spread_engine.update_spread_pos(spread)

        event: Event = Event(EVENT_SPREAD_POS, spread.get_item())
        self.event_engine.put(event)

    def get_leg(self, vt_symbol: str) -> LegData:
        """"""
        leg: LegData = self.legs.get(vt_symbol, None)

        if not leg:
            leg = LegData(vt_symbol)
            self.legs[vt_symbol] = leg

            # Subscribe market data
            contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
            if contract:
                leg.update_contract(contract)

                req: SubscribeRequest = SubscribeRequest(
                    contract.symbol,
                    contract.exchange
                )
                self.main_engine.subscribe(req, contract.gateway_name)

            # Initialize leg position
            positions: List[PositionData] = self.main_engine.get_all_positions()
            for position in positions:
                if position.vt_symbol == vt_symbol:
                    leg.update_position(position)

        return leg

    def add_spread(
        self,
        name: str,
        leg_settings: List[Dict],
        price_formula: str,
        active_symbol: str,
        min_volume: float,
        save: bool = True
    ) -> None:
        """"""
        if name in self.spreads:
            self.write_log("价差创建失败，名称重复：{}".format(name))
            return

        legs: List[LegData] = []
        variable_symbols: Dict[str, str] = {}
        variable_directions: Dict[str, int] = {}
        trading_multipliers: Dict[str, int] = {}

        for leg_setting in leg_settings:
            vt_symbol: str = leg_setting["vt_symbol"]
            variable: str = leg_setting["variable"]
            leg: LegData = self.get_leg(vt_symbol)

            legs.append(leg)
            variable_symbols[variable] = vt_symbol
            variable_directions[variable] = leg_setting["trading_direction"]
            trading_multipliers[vt_symbol] = leg_setting["trading_multiplier"]

        spread: SpreadData = SpreadData(
            name,
            legs,
            variable_symbols,
            variable_directions,
            price_formula,
            trading_multipliers,
            active_symbol,
            min_volume
        )
        self.spreads[name] = spread

        for leg in spread.legs.values():
            self.symbol_spread_map[leg.vt_symbol].append(spread)

        if save:
            self.save_setting()

        self.write_log("价差创建成功：{}".format(name))
        self.put_data_event(spread)

    def remove_spread(self, name: str) -> None:
        """"""
        if name not in self.spreads:
            return

        spread: SpreadData = self.spreads.pop(name)

        for leg in spread.legs.values():
            self.symbol_spread_map[leg.vt_symbol].remove(spread)

        self.save_setting()
        self.write_log("价差移除成功：{}，重启后生效".format(name))

    def get_spread(self, name: str) -> Optional[SpreadData]:
        """"""
        spread: SpreadData = self.spreads.get(name, None)
        return spread

    def get_all_spread_names(self) -> List[str]:
        """"""
        return list(self.spreads.keys())

    def update_order_spread_map(self, vt_orderid: str, spread: SpreadData) -> None:
        """更新委托号对应的价差映射关系"""
        self.order_spread_map[vt_orderid] = spread


class SpreadAlgoEngine:
    """"""
    algo_class: SpreadTakerAlgo = SpreadTakerAlgo

    def __init__(self, spread_engine: SpreadEngine) -> None:
        """"""
        self.spread_engine: SpreadEngine = spread_engine
        self.data_engine: SpreadDataEngine = spread_engine.data_engine
        self.main_engine: MainEngine = spread_engine.main_engine
        self.event_engine: EventEngine = spread_engine.event_engine

        self.write_log = spread_engine.write_log

        self.spreads: Dict[str, SpreadData] = {}
        self.algos: Dict[str, SpreadAlgoTemplate] = {}

        self.order_algo_map: Dict[str, SpreadAlgoTemplate] = {}
        self.symbol_algo_map: Dict[str, List[SpreadAlgoTemplate]] = defaultdict(list)

        self.algo_count: int = 0
        self.vt_tradeids: set = set()

    def start(self) -> None:
        """"""
        self.register_event()

        self.write_log("价差算法引擎启动成功")

    def stop(self) -> None:
        """"""
        for algo in self.algos.values():
            self.stop_algo(algo)

    def register_event(self) -> None:
        """"""
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)

    def update_spread_data(self, spread: SpreadData) -> None:
        """"""
        self.spreads[spread.name] = spread

    def process_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data
        algos: List[SpreadAlgoTemplate] = self.symbol_algo_map[tick.vt_symbol]
        if not algos:
            return

        buf: List[SpreadAlgoTemplate] = copy(algos)
        for algo in buf:
            if not algo.is_active():
                algos.remove(algo)
            else:
                algo.update_tick(tick)

    def process_order_event(self, event: Event) -> None:
        """"""
        order: OrderData = event.data

        algo: SpreadAlgoTemplate = self.order_algo_map.get(order.vt_orderid, None)
        if algo and algo.is_active():
            algo.update_order(order)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data

        # Filter duplicate trade push
        if trade.vt_tradeid in self.vt_tradeids:
            return
        self.vt_tradeids.add(trade.vt_tradeid)

        algo: SpreadAlgoTemplate = self.order_algo_map.get(trade.vt_orderid, None)
        if algo and algo.is_active():
            algo.update_trade(trade)

    def process_timer_event(self, event: Event) -> None:
        """"""
        buf: List[SpreadAlgoTemplate] = list(self.algos.values())

        for algo in buf:
            if not algo.is_active():
                self.algos.pop(algo.algoid)
            else:
                algo.update_timer()

    def start_algo(
        self,
        spread_name: str,
        direction: Direction,
        price: float,
        volume: float,
        payup: int,
        interval: int,
        lock: bool,
        extra: dict
    ) -> str:
        # Find spread object
        spread: SpreadData = self.spreads.get(spread_name, None)
        if not spread:
            self.write_log("创建价差算法失败，找不到价差：{}".format(spread_name))
            return ""

        # Generate algoid str
        self.algo_count += 1
        algo_count_str: str = str(self.algo_count).rjust(6, "0")
        algoid: str = f"{self.algo_class.algo_name}_{algo_count_str}"

        # Create algo object
        algo: SpreadAlgoTemplate = self.algo_class(
            self,
            algoid,
            spread,
            direction,
            price,
            volume,
            payup,
            interval,
            lock,
            extra
        )
        self.algos[algoid] = algo

        # Generate map between vt_symbol and algo
        for leg in spread.legs.values():
            self.symbol_algo_map[leg.vt_symbol].append(algo)

        # Put event to update GUI
        self.put_algo_event(algo)

        return algoid

    def stop_algo(
        self,
        algoid: str
    ) -> None:
        """"""
        algo: SpreadAlgoTemplate = self.algos.get(algoid, None)
        if not algo:
            self.write_log("停止价差算法失败，找不到算法：{}".format(algoid))
            return

        algo.stop()

    def put_algo_event(self, algo: SpreadAlgoTemplate) -> None:
        """"""
        self.spread_engine.update_spread_algo(algo)

        event: Event = Event(EVENT_SPREAD_ALGO, algo.get_item())
        self.event_engine.put(event)

    def write_algo_log(self, algo: SpreadAlgoTemplate, msg: str) -> None:
        """"""
        msg: str = f"{algo.algoid}：{msg}"
        self.write_log(msg)

    def send_order(
        self,
        algo: SpreadAlgoTemplate,
        vt_symbol: str,
        price: float,
        volume: float,
        direction: Direction,
        lock: bool,
        fak: bool
    ) -> List[str]:
        """"""
        # 创建原始委托请求
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        if fak:
            order_type: OrderType = OrderType.FAK
        else:
            order_type: OrderType = OrderType.LIMIT

        original_req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            offset=Offset.OPEN,
            type=order_type,
            price=price,
            volume=volume,
            reference=f"{APP_NAME}_{algo.spread_name}"
        )

        # 判断使用净仓还是锁仓模式
        net: bool = not lock

        # 执行委托转换
        req_list: List[OrderRequest] = self.main_engine.convert_order_request(
            original_req,
            contract.gateway_name,
            lock,
            net
        )

        # Send Orders
        vt_orderids: list = []

        for req in req_list:
            vt_orderid: str = self.main_engine.send_order(
                req, contract.gateway_name)

            # Check if sending order successful
            if not vt_orderid:
                continue

            vt_orderids.append(vt_orderid)

            self.main_engine.update_order_request(req, vt_orderid, contract.gateway_name)

            # Save relationship between orderid and algo.
            self.order_algo_map[vt_orderid] = algo

            # 将委托号和价差的关系缓存下来
            self.data_engine.update_order_spread_map(vt_orderid, algo.spread)

        return vt_orderids

    def cancel_order(self, algo: SpreadAlgoTemplate, vt_orderid: str) -> None:
        """"""
        order: Optional[OrderData] = self.main_engine.get_order(vt_orderid)
        if not order:
            self.write_algo_log(algo, "撤单失败，找不到委托{}".format(vt_orderid))
            return

        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def get_tick(self, vt_symbol: str) -> Optional[TickData]:
        """"""
        return self.main_engine.get_tick(vt_symbol)

    def get_contract(self, vt_symbol: str) -> Optional[ContractData]:
        """"""
        return self.main_engine.get_contract(vt_symbol)


class SpreadStrategyEngine:
    """"""

    engine_type: EngineType = EngineType.LIVE

    setting_filename: str = "spread_trading_strategy.json"

    def __init__(self, spread_engine: SpreadEngine) -> None:
        """"""
        self.spread_engine: SpreadEngine = spread_engine
        self.main_engine: MainEngine = spread_engine.main_engine
        self.event_engine: EventEngine = spread_engine.event_engine

        self.write_log = spread_engine.write_log

        self.strategy_setting: dict = {}

        self.classes: dict = {}
        self.strategies: Dict[str, SpreadStrategyTemplate] = {}

        self.order_strategy_map: Dict[str, SpreadStrategyTemplate] = {}
        self.algo_strategy_map: Dict[str, SpreadStrategyTemplate] = {}
        self.spread_strategy_map: Dict[str, List[SpreadStrategyTemplate]] = defaultdict(list)

        self.init_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

        self.vt_tradeids: set = set()

        self.load_strategy_class()

    def start(self) -> None:
        """"""
        self.load_strategy_setting()
        self.register_event()

        self.write_log("价差策略引擎启动成功")

    def close(self) -> None:
        """"""
        self.stop_all_strategies()

    def load_strategy_class(self) -> None:
        """
        Load strategy class from source code.
        """
        path1: Path = Path(__file__).parent.joinpath("strategies")
        self.load_strategy_class_from_folder(path1, "vnpy_spreadtrading.strategies")

        path2: Path = Path.cwd().joinpath("strategies")
        self.load_strategy_class_from_folder(path2, "strategies")

    def load_strategy_class_from_folder(self, path: Path, module_name: str = "") -> None:
        """
        Load strategy class from certain folder.
        """
        for dirpath, dirnames, filenames in os.walk(str(path)):
            for filename in filenames:
                if filename.split(".")[-1] in ("py", "pyd", "so"):
                    strategy_module_name = ".".join([module_name, filename.split(".")[0]])
                    self.load_strategy_class_from_module(strategy_module_name)

    def load_strategy_class_from_module(self, module_name: str) -> None:
        """
        Load strategy class from module file.
        """
        try:
            module: ModuleType = importlib.import_module(module_name)

            for name in dir(module):
                value = getattr(module, name)
                if (isinstance(value, type) and issubclass(value, SpreadStrategyTemplate) and value is not SpreadStrategyTemplate):
                    self.classes[value.__name__] = value
        except:  # noqa
            msg: str = f"策略文件{module_name}加载失败，触发异常：\n{traceback.format_exc()}"
            self.write_log(msg)

    def get_all_strategy_class_names(self) -> list:
        """"""
        return list(self.classes.keys())

    def load_strategy_setting(self) -> None:
        """
        Load setting file.
        """
        self.strategy_setting = load_json(self.setting_filename)

        for strategy_name, strategy_config in self.strategy_setting.items():
            self.add_strategy(
                strategy_config["class_name"],
                strategy_name,
                strategy_config["spread_name"],
                strategy_config["setting"]
            )

    def update_strategy_setting(self, strategy_name: str, setting: dict) -> None:
        """
        Update setting file.
        """
        strategy: SpreadStrategyTemplate = self.strategies[strategy_name]

        self.strategy_setting[strategy_name] = {
            "class_name": strategy.__class__.__name__,
            "spread_name": strategy.spread_name,
            "setting": setting,
        }
        save_json(self.setting_filename, self.strategy_setting)

    def remove_strategy_setting(self, strategy_name: str) -> None:
        """
        Update setting file.
        """
        if strategy_name not in self.strategy_setting:
            return

        self.strategy_setting.pop(strategy_name)
        save_json(self.setting_filename, self.strategy_setting)

    def register_event(self) -> None:
        """"""
        ee: EventEngine = self.event_engine
        ee.register(EVENT_ORDER, self.process_order_event)
        ee.register(EVENT_TRADE, self.process_trade_event)

    def update_spread_data(self, spread: SpreadData) -> None:
        """"""
        strategies: List[SpreadStrategyTemplate] = self.spread_strategy_map[spread.name]

        for strategy in strategies:
            if strategy.inited:
                self.call_strategy_func(strategy, strategy.on_spread_data)

    def update_spread_pos(self, spread: SpreadData) -> None:
        """"""
        strategies: List[SpreadStrategyTemplate] = self.spread_strategy_map[spread.name]

        for strategy in strategies:
            if strategy.inited:
                self.call_strategy_func(strategy, strategy.on_spread_pos)

    def update_spread_algo(self, algo: SpreadAlgoTemplate) -> None:
        """"""
        strategy: SpreadStrategyTemplate = self.algo_strategy_map.get(algo.algoid, None)

        if strategy:
            self.call_strategy_func(
                strategy, strategy.update_spread_algo, algo)

    def process_order_event(self, event: Event) -> None:
        """"""
        order: OrderData = event.data
        strategy: SpreadStrategyTemplate = self.order_strategy_map.get(order.vt_orderid, None)

        if strategy:
            self.call_strategy_func(strategy, strategy.update_order, order)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data
        strategy: SpreadStrategyTemplate = self.order_strategy_map.get(trade.vt_orderid, None)

        if strategy:
            self.call_strategy_func(strategy, strategy.on_trade, trade)

    def call_strategy_func(
        self, strategy: SpreadStrategyTemplate, func: Callable, params: Any = None
    ) -> None:
        """
        Call function of a strategy and catch any exception raised.
        """
        try:
            if params:
                func(params)
            else:
                func()
        except Exception:
            strategy.trading = False
            strategy.inited = False

            msg: str = f"触发异常已停止\n{traceback.format_exc()}"
            self.write_strategy_log(strategy, msg)

    def add_strategy(
        self, class_name: str, strategy_name: str, spread_name: str, setting: dict
    ) -> None:
        """
        Add a new strategy.
        """
        if strategy_name in self.strategies:
            self.write_log(f"创建策略失败，存在重名{strategy_name}")
            return

        strategy_class: type = self.classes.get(class_name, None)
        if not strategy_class:
            self.write_log(f"创建策略失败，找不到策略类{class_name}")
            return

        spread: Optional[SpreadData] = self.spread_engine.get_spread(spread_name)
        if not spread:
            self.write_log(f"创建策略失败，找不到价差{spread_name}")
            return

        strategy: SpreadStrategyTemplate = strategy_class(self, strategy_name, spread, setting)
        self.strategies[strategy_name] = strategy

        # Add vt_symbol to strategy map.
        strategies: List[SpreadStrategyTemplate] = self.spread_strategy_map[spread_name]
        strategies.append(strategy)

        # Update to setting file.
        self.update_strategy_setting(strategy_name, setting)

        self.put_strategy_event(strategy)

    def edit_strategy(self, strategy_name: str, setting: dict) -> None:
        """
        Edit parameters of a strategy.
        """
        strategy: SpreadStrategyTemplate = self.strategies[strategy_name]
        strategy.update_setting(setting)

        self.update_strategy_setting(strategy_name, setting)
        self.put_strategy_event(strategy)

    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy.
        """
        strategy: SpreadStrategyTemplate = self.strategies[strategy_name]
        if strategy.trading:
            self.write_log(f"策略{strategy.strategy_name}移除失败，请先停止")
            return

        # Remove setting
        self.remove_strategy_setting(strategy_name)

        # Remove from symbol strategy map
        strategies: List[SpreadStrategyTemplate] = self.spread_strategy_map[strategy.spread_name]
        strategies.remove(strategy)

        # Remove from strategies
        self.strategies.pop(strategy_name)

        return True

    def init_strategy(self, strategy_name: str) -> Future:
        """"""
        return self.init_executor.submit(self._init_strategy, strategy_name)

    def _init_strategy(self, strategy_name: str) -> None:
        """"""
        strategy: SpreadStrategyTemplate = self.strategies[strategy_name]

        if strategy.inited:
            self.write_log(f"{strategy_name}已经完成初始化，禁止重复操作")
            return

        self.call_strategy_func(strategy, strategy.on_init)
        strategy.inited = True

        self.put_strategy_event(strategy)
        self.write_log(f"{strategy_name}初始化完成")

    def start_strategy(self, strategy_name: str) -> None:
        """"""
        strategy: SpreadStrategyTemplate = self.strategies[strategy_name]
        if not strategy.inited:
            self.write_log(f"策略{strategy.strategy_name}启动失败，请先初始化")
            return

        if strategy.trading:
            self.write_log(f"{strategy_name}已经启动，请勿重复操作")
            return

        self.call_strategy_func(strategy, strategy.on_start)
        strategy.trading = True

        self.put_strategy_event(strategy)

    def stop_strategy(self, strategy_name: str) -> None:
        """"""
        strategy: SpreadStrategyTemplate = self.strategies[strategy_name]
        if not strategy.trading:
            return

        self.call_strategy_func(strategy, strategy.on_stop)

        strategy.stop_all_algos()
        strategy.cancel_all_orders()

        strategy.trading = False

        self.put_strategy_event(strategy)

    def init_all_strategies(self) -> None:
        """"""
        for strategy in self.strategies.keys():
            self.init_strategy(strategy)

    def start_all_strategies(self) -> None:
        """"""
        for strategy in self.strategies.keys():
            self.start_strategy(strategy)

    def stop_all_strategies(self) -> None:
        """"""
        for strategy in self.strategies.keys():
            self.stop_strategy(strategy)

    def get_strategy_class_parameters(self, class_name: str) -> dict:
        """
        Get default parameters of a strategy class.
        """
        strategy_class: type = self.classes[class_name]

        parameters: dict = {}
        for name in strategy_class.parameters:
            parameters[name] = getattr(strategy_class, name)

        return parameters

    def get_strategy_parameters(self, strategy_name) -> dict:
        """
        Get parameters of a strategy.
        """
        strategy: SpreadStrategyTemplate = self.strategies[strategy_name]
        return strategy.get_parameters()

    def start_algo(
        self,
        strategy: SpreadStrategyTemplate,
        spread_name: str,
        direction: Direction,
        price: float,
        volume: float,
        payup: int,
        interval: int,
        lock: bool,
        extra: dict
    ) -> str:
        """"""
        algoid: str = self.spread_engine.start_algo(
            spread_name,
            direction,
            price,
            volume,
            payup,
            interval,
            lock,
            extra
        )

        self.algo_strategy_map[algoid] = strategy

        return algoid

    def stop_algo(self, strategy: SpreadStrategyTemplate, algoid: str) -> None:
        """"""
        self.spread_engine.stop_algo(algoid)

    def stop_all_algos(self, strategy: SpreadStrategyTemplate) -> None:
        """"""
        pass

    def send_order(
        self,
        strategy: SpreadStrategyTemplate,
        vt_symbol: str,
        price: float,
        volume: float,
        direction: Direction,
        offset: Offset,
        lock: bool
    ) -> List[str]:
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        original_req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            offset=offset,
            type=OrderType.LIMIT,
            price=price,
            volume=volume,
            reference=f"{APP_NAME}_{strategy.strategy_name}"
        )

        # Convert with offset converter
        req_list: List[OrderRequest] = self.main_engine.convert_order_request(
            original_req,
            contract.gateway_name,
            lock
        )

        # Send Orders
        vt_orderids: list = []

        for req in req_list:
            vt_orderid: str = self.main_engine.send_order(
                req, contract.gateway_name)

            # Check if sending order successful
            if not vt_orderid:
                continue

            vt_orderids.append(vt_orderid)

            self.main_engine.update_order_request(req, vt_orderid, contract.gateway_name)

            # Save relationship between orderid and strategy.
            self.order_strategy_map[vt_orderid] = strategy

        return vt_orderids

    def cancel_order(self, strategy: SpreadStrategyTemplate, vt_orderid: str) -> None:
        """"""
        order: Optional[OrderData] = self.main_engine.get_order(vt_orderid)
        if not order:
            self.write_strategy_log(
                strategy, "撤单失败，找不到委托{}".format(vt_orderid))
            return

        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def cancel_all_orders(self, strategy: SpreadStrategyTemplate) -> None:
        """"""
        pass

    def put_strategy_event(self, strategy: SpreadStrategyTemplate) -> None:
        """"""
        data: dict = strategy.get_data()
        event: Event = Event(EVENT_SPREAD_STRATEGY, data)
        self.event_engine.put(event)

    def write_strategy_log(self, strategy: SpreadStrategyTemplate, msg: str) -> None:
        """"""
        msg: str = f"{strategy.strategy_name}：{msg}"
        self.write_log(msg)

    def send_email(self, msg: str, strategy: SpreadStrategyTemplate = None) -> None:
        """"""
        if strategy:
            subject: str = f"{strategy.strategy_name}"
        else:
            subject: str = "价差策略引擎"

        self.main_engine.send_email(subject, msg)

    def get_engine_type(self) -> EngineType:
        """"""
        return self.engine_type

    def load_bar(
        self, spread: SpreadData, days: int, interval: Interval, callback: Callable
    ) -> None:
        """"""
        end: datetime = datetime.now(DB_TZ)
        start: datetime = end - timedelta(days)

        bars: List[BarData] = load_bar_data(spread, interval, start, end, output=self.write_log)

        for bar in bars:
            callback(bar)

    def load_tick(self, spread: SpreadData, days: int, callback: Callable) -> None:
        """"""
        end: datetime = datetime.now(DB_TZ)
        start: datetime = end - timedelta(days)

        ticks: List[TickData] = load_tick_data(spread, start, end)

        for tick in ticks:
            callback(tick)
