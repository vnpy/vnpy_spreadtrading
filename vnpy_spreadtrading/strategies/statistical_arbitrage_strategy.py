from vnpy.trader.utility import BarGenerator, ArrayManager
from vnpy_spreadtrading import (
    SpreadStrategyTemplate,
    SpreadAlgoTemplate,
    TickData,
    BarData
)


class StatisticalArbitrageStrategy(SpreadStrategyTemplate):
    """"""

    author = "用Python的交易员"

    boll_window = 20
    boll_dev = 2
    max_pos = 10
    payup = 10
    interval = 5

    spread_pos = 0.0
    boll_up = 0.0
    boll_down = 0.0
    boll_mid = 0.0

    parameters = [
        "boll_window",
        "boll_dev",
        "max_pos",
        "payup",
        "interval"
    ]
    variables = [
        "spread_pos",
        "boll_up",
        "boll_down",
        "boll_mid"
    ]

    def on_init(self) -> None:
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

        self.bg = BarGenerator(self.on_spread_bar)
        self.am = ArrayManager()

        self.load_bar(10)

    def on_start(self) -> None:
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

        self.put_event()

    def on_spread_data(self) -> None:
        """
        Callback when spread price is updated.
        """
        tick = self.get_spread_tick()
        self.on_spread_tick(tick)

    def on_spread_tick(self, tick: TickData) -> None:
        """
        Callback when new spread tick data is generated.
        """
        self.bg.update_tick(tick)

    def on_spread_bar(self, bar: BarData) -> None:
        """
        Callback when spread bar data is generated.
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
        Callback when spread position is updated.
        """
        self.spread_pos = self.get_spread_pos()
        self.put_event()

    def on_spread_algo(self, algo: SpreadAlgoTemplate) -> None:
        """
        Callback when algo status is updated.
        """
        pass
