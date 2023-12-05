import traceback
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Callable, Type, Dict, List, Optional
from functools import partial

import numpy as np
from pandas import DataFrame
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vnpy.trader.constant import (
    Direction,
    Offset,
    Exchange,
    Interval,
    Status
)
from vnpy.trader.object import TradeData, BarData, TickData
from vnpy.trader.optimize import (
    OptimizationSetting,
    check_optimization_setting,
    run_bf_optimization,
    run_ga_optimization
)

from .template import SpreadStrategyTemplate, SpreadAlgoTemplate
from .base import (
    SpreadData,
    BacktestingMode,
    load_bar_data,
    load_tick_data,
    EngineType
)


INTERVAL_DELTA_MAP: Dict[Interval, timedelta] = {
    Interval.TICK: timedelta(milliseconds=1),
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}


class BacktestingEngine:
    """"""

    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self) -> None:
        """"""
        self.spread: SpreadData = None

        self.start: datetime = None
        self.end: datetime = None
        self.rate: float = 0
        self.slippage: float = 0
        self.size: float = 1
        self.pricetick: float = 0
        self.capital: int = 1_000_000
        self.risk_free: float = 0
        self.annual_days: int = 240
        self.mode: BacktestingMode = BacktestingMode.BAR

        self.strategy_class: Type[SpreadStrategyTemplate] = None
        self.strategy: SpreadStrategyTemplate = None
        self.tick: TickData = None
        self.bar: BarData = None
        self.datetime: datetime = None

        self.interval: Interval = None
        self.days: int = 0
        self.callback: Callable = None
        self.history_data: list = []

        self.algo_count: int = 0
        self.algos: Dict[str, SpreadAlgoTemplate] = {}
        self.active_algos: Dict[str, SpreadAlgoTemplate] = {}

        self.trade_count: int = 0
        self.trades: Dict[str, TradeData] = {}

        self.logs: list = []

        self.daily_results: Dict[date, DailyResult] = {}
        self.daily_df: DataFrame = None

    def output(self, msg) -> None:
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def clear_data(self) -> None:
        """
        Clear all data of last backtesting.
        """
        self.strategy = None
        self.tick = None
        self.bar = None
        self.datetime = None

        self.algo_count = 0
        self.algos.clear()
        self.active_algos.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

    def set_parameters(
        self,
        spread: SpreadData,
        interval: Interval,
        start: datetime,
        rate: float,
        slippage: float,
        size: float,
        pricetick: float,
        capital: int = 0,
        risk_free: float = 0,
        annual_days: int = 240,
        end: datetime = None,
        mode: BacktestingMode = BacktestingMode.BAR
    ) -> None:
        """"""
        self.spread = spread
        self.interval = Interval(interval)
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.start = start
        self.capital = capital
        self.risk_free = risk_free
        self.annual_days = annual_days
        self.end = end
        self.mode = mode

    def add_strategy(self, strategy_class: type, setting: dict) -> None:
        """"""
        self.strategy_class = strategy_class

        self.strategy = strategy_class(
            self,
            strategy_class.__name__,
            self.spread,
            setting
        )

    def load_data(self) -> None:
        """"""
        self.output("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        if self.mode == BacktestingMode.BAR:
            self.history_data = load_bar_data(
                spread=self.spread,
                interval=self.interval,
                start=self.start,
                end=self.end,
                pricetick=self.pricetick,
                backtesting=True
            )
        else:
            self.history_data = load_tick_data(
                self.spread,
                self.start,
                self.end
            )

        self.output(f"历史数据加载完成，数据量：{len(self.history_data)}")

    def run_backtesting(self) -> None:
        """"""
        if self.mode == BacktestingMode.BAR:
            func = self.new_bar
        else:
            func = self.new_tick

        self.strategy.on_init()
        self.strategy.inited = True
        self.output("策略初始化完成")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("开始回放历史数据")

        for data in self.history_data:
            try:
                func(data)
            except Exception:
                self.output("触发异常，回测终止")
                self.output(traceback.format_exc())
                return

        self.output("历史数据回放结束")

    def calculate_result(self) -> DataFrame:
        """"""
        self.output("开始计算逐日盯市盈亏")

        if not self.trades:
            self.output("回测成交记录为空")

        # Add trade data into daily reuslt.
        for trade in self.trades.values():
            d: date = trade.datetime.date()
            daily_result = self.daily_results[d]
            daily_result.add_trade(trade)

        # Calculate daily result by iteration.
        pre_close = 0
        start_pos = 0

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_close,
                start_pos,
                self.size,
                self.rate,
                self.slippage
            )

            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

        # Generate dataframe
        results: defaultdict = defaultdict(list)

        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)

        if results:
            self.daily_df: DataFrame = DataFrame.from_dict(results).set_index("date")

        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True) -> dict:
        """"""
        self.output("开始计算策略统计指标")

        # Check DataFrame input exterior
        if df is None:
            df: DataFrame = self.daily_df

        # Init all statistics default value
        start_date: str = ""
        end_date: str = ""
        total_days: int = 0
        profit_days: int = 0
        loss_days: int = 0
        end_balance: float = 0
        max_drawdown: float = 0
        max_ddpercent: float = 0
        max_drawdown_duration: int = 0
        total_net_pnl: float = 0
        daily_net_pnl: float = 0
        total_commission: float = 0
        daily_commission: float = 0
        total_slippage: float = 0
        daily_slippage: float = 0
        total_turnover: float = 0
        daily_turnover: float = 0
        total_trade_count: int = 0
        daily_trade_count: int = 0
        total_return: float = 0
        annual_return: float = 0
        daily_return: float = 0
        return_std: float = 0
        sharpe_ratio: float = 0
        return_drawdown_ratio: float = 0

        # Check if balance is always positive
        positive_balance: bool = False

        # Check for init DataFrame
        if df is not None:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # All balance value needs to be positive
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                self.output("回测中出现爆仓（资金小于等于0），无法计算策略统计指标")

        if positive_balance:
            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days: int = len(df)
            profit_days: int = len(df[df["net_pnl"] > 0])
            loss_days: int = len(df[df["net_pnl"] < 0])

            end_balance: float = df["balance"].iloc[-1]
            max_drawdown: float = df["drawdown"].min()
            max_ddpercent: float = df["ddpercent"].min()
            max_drawdown_end: float = df["drawdown"].idxmin()
            max_drawdown_start: float = df["balance"][:max_drawdown_end].idxmax()
            max_drawdown_duration: int = (max_drawdown_end - max_drawdown_start).days

            total_net_pnl: float = df["net_pnl"].sum()
            daily_net_pnl: float = total_net_pnl / total_days

            total_commission: float = df["commission"].sum()
            daily_commission: float = total_commission / total_days

            total_slippage: float = df["slippage"].sum()
            daily_slippage: float = total_slippage / total_days

            total_turnover: float = df["turnover"].sum()
            daily_turnover: float = total_turnover / total_days

            total_trade_count: int = df["trade_count"].sum()
            daily_trade_count: int = total_trade_count / total_days

            total_return: float = (end_balance / self.capital - 1) * 100
            annual_return: float = total_return / total_days * self.annual_days
            daily_return: float = df["return"].mean() * 100
            return_std: float = df["return"].std() * 100

            if return_std:
                daily_risk_free: float = self.risk_free / self.annual_days
                sharpe_ratio: float = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days)
            else:
                sharpe_ratio: float = 0

            return_drawdown_ratio: float = -total_return / max_ddpercent

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"首个交易日：\t{start_date}")
            self.output(f"最后交易日：\t{end_date}")

            self.output(f"总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")
            self.output(f"最长回撤天数: \t{max_drawdown_duration}")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")

        statistics: dict = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        return statistics

    def show_chart(self, df: DataFrame = None) -> None:
        """"""
        # Check DataFrame input exterior
        if df is None:
            df: DataFrame = self.daily_df

        # Check for init DataFrame
        if df is None:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def run_bf_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output=True,
        max_workers: int = None
    ) -> list:
        """"""
        if not check_optimization_setting(optimization_setting):
            return

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_bf_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            output=self.output,
        )

        if output:
            for result in results:
                msg: str = f"参数：{result[0]}, 目标：{result[1]}"
                self.output(msg)

        return results

    run_optimization = run_bf_optimization

    def run_ga_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output=True,
        max_workers: int = None,
        ngen_size: int = 30
    ) -> list:
        """"""
        if not check_optimization_setting(optimization_setting):
            return

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_ga_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            ngen_size=ngen_size,
            output=self.output
        )

        if output:
            for result in results:
                msg: str = f"参数：{result[0]}, 目标：{result[1]}"
                self.output(msg)

        return results

    def update_daily_close(self, price: float) -> None:
        """"""
        d: date = self.datetime.date()

        daily_result: Optional[DailyResult] = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
        else:
            self.daily_results[d] = DailyResult(d, price)

    def new_bar(self, bar: BarData) -> None:
        """"""
        self.bar = bar
        self.datetime = bar.datetime
        self.cross_algo()

        self.strategy.on_spread_bar(bar)

        self.update_daily_close(bar.close_price)

    def new_tick(self, tick: TickData) -> None:
        """"""
        self.tick = tick
        self.datetime = tick.datetime
        self.cross_algo()

        self.spread.bid_price = tick.bid_price_1
        self.spread.bid_volume = tick.bid_volume_1
        self.spread.ask_price = tick.ask_price_1
        self.spread.ask_volume = tick.ask_volume_1
        self.spread.datetime = tick.datetime

        self.strategy.on_spread_data()

        self.update_daily_close(tick.last_price)

    def cross_algo(self) -> None:
        """
        Cross limit order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.close_price
            short_cross_price = self.bar.close_price
        else:
            long_cross_price = self.tick.ask_price_1
            short_cross_price = self.tick.bid_price_1

        for algo in list(self.active_algos.values()):
            # Check whether limit orders can be filled.
            long_cross: bool = (
                algo.direction == Direction.LONG
                and algo.price >= long_cross_price
            )

            short_cross: bool = (
                algo.direction == Direction.SHORT
                and algo.price <= short_cross_price
            )

            if not long_cross and not short_cross:
                continue

            # Push order udpate with status "all traded" (filled).
            algo.traded = algo.target
            algo.traded_volume = algo.volume
            algo.traded_price = algo.price
            algo.status = Status.ALLTRADED
            self.strategy.update_spread_algo(algo)

            self.active_algos.pop(algo.algoid)

            # Push trade update
            self.trade_count += 1

            if long_cross:
                trade_price = long_cross_price
                pos_change = algo.volume
            else:
                trade_price = short_cross_price
                pos_change = -algo.volume

            trade: TradeData = TradeData(
                symbol=self.spread.name,
                exchange=Exchange.LOCAL,
                orderid=algo.algoid,
                tradeid=str(self.trade_count),
                direction=algo.direction,
                price=trade_price,
                volume=algo.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            if self.mode == BacktestingMode.BAR:
                trade.value = self.bar.value
            else:
                trade.value = trade_price

            self.spread.net_pos += pos_change
            self.strategy.on_spread_pos()

            self.trades[trade.vt_tradeid] = trade

    def load_bar(
        self, spread: SpreadData, days: int, interval: Interval, callback: Callable
    ) -> None:
        """"""
        self.callback = callback

        init_end = self.start - INTERVAL_DELTA_MAP[interval]
        init_start = self.start - timedelta(days=days)

        bars: List[BarData] = load_bar_data(
            spread=self.spread,
            interval=self.interval,
            start=init_start,
            end=init_end,
            pricetick=self.pricetick,
            backtesting=True
        )

        for bar in bars:
            callback(bar)

        return bars

    def load_tick(self, spread: SpreadData, days: int, callback: Callable) -> None:
        """"""
        self.days = days

        init_end = self.start - INTERVAL_DELTA_MAP[Interval.TICK]
        init_start = self.start - timedelta(days=days)

        ticks: List[TickData] = load_tick_data(
            self.spread,
            init_start,
            init_end
        )

        for tick in ticks:
            callback(callback)

        return ticks

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
        self.algo_count += 1
        algoid: str = str(self.algo_count)

        algo: SpreadAlgoTemplate = SpreadAlgoTemplate(
            self,
            algoid,
            self.spread,
            direction,
            price,
            volume,
            payup,
            interval,
            lock,
            extra
        )

        self.algos[algoid] = algo
        self.active_algos[algoid] = algo

        return algoid

    def stop_algo(
        self,
        strategy: SpreadStrategyTemplate,
        algoid: str
    ) -> None:
        """"""
        if algoid not in self.active_algos:
            return
        algo: SpreadAlgoTemplate = self.active_algos.pop(algoid)

        algo.status = Status.CANCELLED
        self.strategy.update_spread_algo(algo)

    def send_order(
        self,
        strategy: SpreadStrategyTemplate,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool,
        lock: bool
    ) -> None:
        """"""
        pass

    def cancel_order(self, strategy: SpreadStrategyTemplate, vt_orderid: str) -> None:
        """
        Cancel order by vt_orderid.
        """
        pass

    def write_strategy_log(self, strategy: SpreadStrategyTemplate, msg: str) -> None:
        """
        Write log message.
        """
        msg: str = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: SpreadStrategyTemplate = None) -> None:
        """
        Send email to default receiver.
        """
        pass

    def get_engine_type(self) -> EngineType:
        """
        Return engine type.
        """
        return self.engine_type

    def put_strategy_event(self, strategy: SpreadStrategyTemplate) -> None:
        """
        Put an event to update strategy status.
        """
        pass

    def write_algo_log(self, algo: SpreadAlgoTemplate, msg: str) -> None:
        """"""
        pass


class DailyResult:
    """"""

    def __init__(self, date: date, close_price: float) -> None:
        """"""
        self.date: date = date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: List[TradeData] = []
        self.trade_count: int = 0

        self.start_pos = 0
        self.end_pos = 0

        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: int,
        rate: float,
        slippage: float
    ) -> None:
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change

            turnover: float = trade.volume * size * trade.value
            self.trading_pnl += pos_change * \
                (self.close_price - trade.price) * size
            self.slippage += trade.volume * size * slippage

            self.turnover += turnover
            self.commission += turnover * rate

        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage


def evaluate(
    target_name: str,
    strategy_class: SpreadStrategyTemplate,
    spread: SpreadData,
    interval: Interval,
    start: datetime,
    rate: float,
    slippage: float,
    size: float,
    pricetick: float,
    capital: int,
    end: datetime,
    setting: dict
) -> tuple:
    """
    Function for running in multiprocessing.pool
    """
    engine: BacktestingEngine = BacktestingEngine()

    engine.set_parameters(
        spread=spread,
        interval=interval,
        start=start,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        end=end,
    )

    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics: dict = engine.calculate_statistics(output=False)

    target_value: float = statistics[target_name]
    return (setting, target_value, statistics)


def wrap_evaluate(engine: BacktestingEngine, target_name: str) -> callable:
    """
    Wrap evaluate function with given setting from backtesting engine.
    """
    func: callable = partial(
        evaluate,
        target_name,
        engine.strategy_class,
        engine.spread,
        engine.interval,
        engine.start,
        engine.rate,
        engine.slippage,
        engine.size,
        engine.pricetick,
        engine.capital,
        engine.end
    )
    return func


def get_target_value(result: list) -> float:
    """
    Get target value for sorting optimization results.
    """
    return result[1]
