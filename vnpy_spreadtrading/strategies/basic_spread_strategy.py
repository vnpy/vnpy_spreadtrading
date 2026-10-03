"""基础价差策略。"""

from datetime import datetime

from vnpy_spreadtrading import (
    SpreadStrategyTemplate,
    SpreadAlgoTemplate,
)


class BasicSpreadStrategy(SpreadStrategyTemplate):
    """在起止时间内按目标价维护开平仓算法。"""

    author = "用Python的交易员"

    buy_price = 0.0
    sell_price = 0.0
    cover_price = 0.0
    short_price = 0.0
    max_pos = 0.0
    payup = 10
    interval = 5
    start_time = "9:00:00"
    end_time = "15:00:00"

    spread_pos = 0.0
    update_time = None
    buy_algoid = ""
    sell_algoid = ""
    short_algoid = ""
    cover_algoid = ""

    parameters = [
        "buy_price",
        "sell_price",
        "cover_price",
        "short_price",
        "max_pos",
        "payup",
        "interval"
    ]
    variables = [
        "spread_pos",
        "update_time",
        "buy_algoid",
        "sell_algoid",
        "short_algoid",
        "cover_algoid",
    ]

    def on_init(self) -> None:
        """
        策略初始化完成时的回调。
        """
        self.write_log("策略初始化")

        self.start_t = datetime.strptime(self.start_time, "%H:%M:%S").time()
        self.end_t = datetime.strptime(self.end_time, "%H:%M:%S").time()

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

        self.update_time = None
        self.buy_algoid = ""
        self.sell_algoid = ""
        self.short_algoid = ""
        self.cover_algoid = ""
        self.put_event()

    def on_spread_data(self) -> None:
        """
        价差价格更新时的回调。
        """
        # Trading is only allowed within given start/end time range
        self.update_time = self.spread.datetime.time()
        if self.update_time < self.start_t or self.update_time >= self.end_t:
            self.stop_open_algos()
            self.stop_close_algos()
            self.put_event()
            return

        self.spread_pos = self.get_spread_pos()

        # No position
        if not self.spread_pos:
            self.stop_close_algos()

            # Start open algos
            if not self.buy_algoid:
                self.buy_algoid = self.start_long_algo(
                    self.buy_price, self.max_pos, self.payup, self.interval
                )

            if not self.short_algoid:
                self.short_algoid = self.start_short_algo(
                    self.short_price, self.max_pos, self.payup, self.interval
                )

        # Long position
        elif self.spread_pos > 0:
            self.stop_open_algos()

            # Start sell close algo
            if not self.sell_algoid:
                self.sell_algoid = self.start_short_algo(
                    self.sell_price, self.spread_pos, self.payup, self.interval
                )

        # Short position
        elif self.spread_pos < 0:
            self.stop_open_algos()

            # Start cover close algo
            if not self.cover_algoid:
                self.cover_algoid = self.start_long_algo(
                    self.cover_price, abs(
                        self.spread_pos), self.payup, self.interval
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
        if not algo.is_active():
            if self.buy_algoid == algo.algoid:
                self.buy_algoid = ""
            elif self.sell_algoid == algo.algoid:
                self.sell_algoid = ""
            elif self.short_algoid == algo.algoid:
                self.short_algoid = ""
            else:
                self.cover_algoid = ""

        self.put_event()

    def stop_open_algos(self) -> None:
        """停止已记录的买入和卖空算法。"""
        if self.buy_algoid:
            self.stop_algo(self.buy_algoid)

        if self.short_algoid:
            self.stop_algo(self.short_algoid)

    def stop_close_algos(self) -> None:
        """停止已记录的卖出和补平算法。"""
        if self.sell_algoid:
            self.stop_algo(self.sell_algoid)

        if self.cover_algoid:
            self.stop_algo(self.cover_algoid)
