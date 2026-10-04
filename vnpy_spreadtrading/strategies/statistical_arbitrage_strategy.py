"""统计套利价差策略。"""

from vnpy.trader.utility import BarGenerator, ArrayManager
from vnpy_spreadtrading import (
    SpreadStrategyTemplate,
    SpreadAlgoTemplate,
    TickData,
    BarData
)


class StatisticalArbitrageStrategy(SpreadStrategyTemplate):
    """用布林带开平价差仓位。"""

    author: str = "用Python的交易员"

    boll_window: int = 20
    boll_dev: int = 2
    max_pos: int = 10
    payup: int = 10
    interval: int = 5

    spread_pos: float = 0.0
    boll_up: float = 0.0
    boll_down: float = 0.0
    boll_mid: float = 0.0

    parameters: list[str] = [
        "boll_window",
        "boll_dev",
        "max_pos",
        "payup",
        "interval"
    ]
    variables: list[str] = [
        "spread_pos",
        "boll_up",
        "boll_down",
        "boll_mid"
    ]

    def on_init(self) -> None:
        """
        策略初始化完成时的回调。
        """
        self.write_log("策略初始化")

        self.bg: BarGenerator = BarGenerator(self.on_spread_bar)
        self.am: ArrayManager = ArrayManager()

        self.load_bar(10)

    def on_start(self) -> None:
        """
        策略启动时的回调。
        """
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """
        策略停止时的回调。
        """
        self.write_log("策略停止")

        self.put_event()

    def on_spread_data(self) -> None:
        """
        价差价格更新时的回调。
        """
        tick: TickData = self.get_spread_tick()
        self.on_spread_tick(tick)

    def on_spread_tick(self, tick: TickData) -> None:
        """
        生成新的价差 Tick 时的回调。
        """
        self.bg.update_tick(tick)

    def on_spread_bar(self, bar: BarData) -> None:
        """
        生成价差 K 线数据时的回调。
        """
        self.stop_all_algos()

        self.am.update_bar(bar)
        if not self.am.inited:
            return

        self.boll_mid = self.am.sma(self.boll_window)
        self.boll_up, self.boll_down = self.am.boll(
            self.boll_window, self.boll_dev)

        if not self.spread_pos:
            if bar.close_price >= self.boll_up:
                self.start_short_algo(
                    bar.close_price - 10,
                    self.max_pos,
                    payup=self.payup,
                    interval=self.interval
                )
            elif bar.close_price <= self.boll_down:
                self.start_long_algo(
                    bar.close_price + 10,
                    self.max_pos,
                    payup=self.payup,
                    interval=self.interval
                )
        elif self.spread_pos < 0:
            if bar.close_price <= self.boll_mid:
                self.start_long_algo(
                    bar.close_price + 10,
                    abs(self.spread_pos),
                    payup=self.payup,
                    interval=self.interval
                )
        else:
            if bar.close_price >= self.boll_mid:
                self.start_short_algo(
                    bar.close_price - 10,
                    abs(self.spread_pos),
                    payup=self.payup,
                    interval=self.interval
                )

        self.put_event()

    def on_spread_pos(self) -> None:
        """
        价差持仓更新时的回调。
        """
        self.spread_pos = self.get_spread_pos()
        self.put_event()

    def on_spread_algo(self, algo: SpreadAlgoTemplate) -> None:
        """
        算法状态更新时的回调。
        """
        pass
