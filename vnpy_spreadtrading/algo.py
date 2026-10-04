"""价差吃单算法。"""

from typing import TYPE_CHECKING

from vnpy.trader.constant import Direction
from vnpy.trader.object import TickData, OrderData, TradeData, ContractData
from vnpy.trader.utility import round_to

from .template import SpreadAlgoTemplate
from .base import SpreadData, LegData

if TYPE_CHECKING:
    from .engine import SpreadAlgoEngine


def round_leg_volume(leg: LegData, volume: float) -> float:
    """有最小交易量时按其取整，否则返回原数量。"""
    if leg.min_volume:
        return round_to(volume, leg.min_volume)
    return volume


class SpreadTakerAlgo(SpreadAlgoTemplate):
    """主动腿吃单并对冲被动腿的算法。"""
    algo_name: str = "SpreadTaker"

    def __init__(
        self,
        algo_engine: "SpreadAlgoEngine",
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
        """调用父类完成初始化。"""
        super().__init__(
            algo_engine,
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

    def on_tick(self, tick: TickData) -> None:
        """委托未结束时返回；未对冲完则对冲被动腿；盘口量为空时返回；价格达到时吃主动腿。"""
        # Return if there are any existing orders
        if not self.is_order_finished():
            return

        # Hedge if active leg is not fully hedged
        if not self.is_hedge_finished():
            self.hedge_passive_legs()
            return

        # Return if tick not inited
        if not self.spread.bid_volume or not self.spread.ask_volume:
            return

        # Otherwise check if should take active leg
        if self.direction == Direction.LONG:
            if self.spread.ask_price <= self.price:
                self.take_active_leg()
        else:
            if self.spread.bid_price >= self.price:
                self.take_active_leg()

    def on_order(self, order: OrderData) -> None:
        """非主动腿委托直接返回；主动腿委托全部结束后，未对冲完则对冲被动腿。"""
        # Only care active leg order update
        if order.vt_symbol != self.spread.active_leg.vt_symbol:
            return

        # Do nothing if still any existing orders
        if not self.is_order_finished():
            return

        # Hedge passive legs if necessary
        if not self.is_hedge_finished():
            self.hedge_passive_legs()

    def on_trade(self, trade: TradeData) -> None:
        """不执行任何操作。"""
        pass

    def on_interval(self) -> None:
        """仍有未完成委托时撤销全部委托。"""
        if not self.is_order_finished():
            self.cancel_all_order()

    def take_active_leg(self) -> None:
        """按剩余目标和盘口量限制主动腿委托数量后发送。"""
        active_symbol: str = self.spread.active_leg.vt_symbol

        # Calculate spread order volume of new round trade
        spread_volume_left: float = self.target - self.traded

        if self.direction == Direction.LONG:
            spread_order_volume: float = self.spread.ask_volume
            spread_order_volume = min(spread_order_volume, spread_volume_left)
        else:
            spread_order_volume = -self.spread.bid_volume
            spread_order_volume = max(spread_order_volume, spread_volume_left)

        # Calculate active leg order volume
        leg_order_volume: float = self.spread.calculate_leg_volume(
            active_symbol,
            spread_order_volume
        )

        # Check active leg volume left
        active_volume_target: float = self.spread.calculate_leg_volume(
            active_symbol,
            self.target
        )
        active_volume_traded: float = self.leg_traded[active_symbol]
        active_volume_left: float = active_volume_target - active_volume_traded

        # Limit order volume to total volume left of the active leg
        if active_volume_left > 0:
            leg_order_volume = min(leg_order_volume, active_volume_left)
        else:
            leg_order_volume= max(leg_order_volume, active_volume_left)

        # Send active leg order
        self.send_leg_order(
            active_symbol,
            leg_order_volume
        )

    def hedge_passive_legs(self) -> None:
        """
        发送委托以对冲全部被动腿。
        """
        # Calcualte spread volume to hedge
        active_leg: LegData = self.spread.active_leg
        active_traded: float = self.leg_traded[active_leg.vt_symbol]
        active_traded = round_leg_volume(active_leg, active_traded)

        hedge_volume: float = self.spread.calculate_spread_volume(
            active_leg.vt_symbol,
            active_traded
        )

        # Calculate passive leg target volume and do hedge
        leg: LegData
        for leg in self.spread.passive_legs:
            passive_traded: float = self.leg_traded[leg.vt_symbol]
            passive_traded = round_leg_volume(leg, passive_traded)

            passive_target: float = self.spread.calculate_leg_volume(
                leg.vt_symbol,
                hedge_volume
            )

            leg_order_volume: float = passive_target - passive_traded
            if leg_order_volume:
                self.send_leg_order(leg.vt_symbol, leg_order_volume)

    def send_leg_order(self, vt_symbol: str, leg_volume: float) -> None:
        """有行情和合约时，正数量按卖一加超价买入，负数量按买一减超价卖出。"""
        leg: LegData = self.spread.legs[vt_symbol]
        leg_tick: TickData | None = self.get_tick(vt_symbol)
        leg_contract: ContractData | None = self.get_contract(vt_symbol)

        if leg_tick and leg_contract:
            if leg_volume > 0:
                price: float = leg_tick.ask_price_1 + leg_contract.pricetick * self.payup
                self.send_order(leg.vt_symbol, price, abs(leg_volume), Direction.LONG)
            elif leg_volume < 0:
                price = leg_tick.bid_price_1 - leg_contract.pricetick * self.payup
                self.send_order(leg.vt_symbol, price, abs(leg_volume), Direction.SHORT)
