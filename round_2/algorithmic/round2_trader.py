from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Tuple
import jsonpickle
import numpy as np
import math

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_vwap = []
        self.resin_prices = []
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_volume: int, sell_volume: int, fair_value: float) -> Tuple[int, int]:
        position_after_take = position + buy_volume - sell_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_volume)
        sell_quantity = position_limit + (position - sell_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
            sell_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
            buy_volume += abs(sent_quantity)
    
        return buy_volume, sell_volume

    def resin_orders(self, order_depth: OrderDepth, fair_value: float, width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_volume = 0
        sell_volume = 0

        # 1) New Volatility-based scaling
        resin_volatility = self.calculate_volatility(self.resin_prices, window=30)
        # Increased base from 20 -> 30, changed from (1 + vol) to (0.5 + vol)
        volatility_scaling = max(10, int(10 / (0.5 + resin_volatility)))

        # 2) Add a threshold around fair_value to avoid small/noisy trades
        threshold = 1.0  # You can tune this number

        # Find the best bid and ask
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1], default=None)
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1], default=None)

        if baaf is None or bbbf is None:
            # If no valid quotes above/below fair_value exist, skip
            return orders

        # Optimize buy orders
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            # Only buy if best_ask < (fair_value - threshold)
            if best_ask < fair_value - threshold:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    order_size = min(quantity, volatility_scaling)
                    orders.append(Order("RAINFOREST_RESIN", best_ask, order_size))
                    buy_volume += order_size

        # Optimize sell orders
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            # Only sell if best_bid > (fair_value + threshold)
            if best_bid > fair_value + threshold:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    order_size = min(quantity, volatility_scaling)
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -order_size))
                    sell_volume += order_size

        # Clear any existing positions
        buy_volume, sell_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN",
            buy_volume, sell_volume, fair_value
        )

        # Adjust buy/sell
        buy_quantity = position_limit - (position + buy_volume)
        sell_quantity = position_limit + (position - sell_volume)

        # Place final buy
        if bbbf is not None and buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))

        # Place final sell
        if baaf is not None and sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))

        return orders
    def kelp_orders(self, order_depth: OrderDepth, timespan: int, width: float, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_volume = 0
        sell_volume = 0

        if order_depth.sell_orders and order_depth.buy_orders:    
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 10]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 10]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.kelp_prices.append(mmmid_price)

            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (-best_bid * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
        
            fair_value = mmmid_price  # Keeping your current choice

            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -order_depth.sell_orders[best_ask]
                quantity = min(ask_amount, position_limit - position)  # Removed 20-unit cap
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_volume += quantity
            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                quantity = min(bid_amount, position_limit + position)  # Removed 20-unit cap
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -quantity))
                    sell_volume += quantity

            # Clear any existing positions
            buy_volume, sell_volume = self.clear_position_order(orders, order_depth, position, position_limit, "KELP", buy_volume, sell_volume, fair_value)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            kelp_volatility = self.calculate_volatility(self.kelp_prices, window=15)
            max_order_size = max(10, int(30 / (1 + kelp_volatility))) 

            buy_quantity = position_limit - (position + buy_volume)
            if buy_quantity > 0:
                order_size = min(buy_quantity, max_order_size)
                orders.append(Order("KELP", bbbf + 1, order_size))

            sell_quantity = position_limit + (position - sell_volume)
            if sell_quantity > 0:
                order_size = min(sell_quantity, max_order_size)
                orders.append(Order("KELP", baaf - 1, -order_size))

        return orders

    def calculate_volatility(self, prices: List[float], window: int) -> float:
        if len(prices) < window:
            return np.std(prices) if prices else 1.0
        return np.std(prices[-window:])

    def run(self, state: TradingState):
        result = {}

        resin_fair = 10000
        resin_width = 2.0
        resin_limit = 50

        kelp_volatility = self.calculate_volatility(self.kelp_prices, window=10)
        kelp_make_width = max(4.0, kelp_volatility * 3.5) 
        kelp_take_width = max(1.5, kelp_volatility * 1.5)
        kelp_limit = 50
        kelp_timespan = 15

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position.get("RAINFOREST_RESIN", 0)
            self.resin_prices.append((min(state.order_depths["RAINFOREST_RESIN"].sell_orders.keys()) + max(state.order_depths["RAINFOREST_RESIN"].buy_orders.keys())) / 2)
            result["RAINFOREST_RESIN"] = self.resin_orders(
                state.order_depths["RAINFOREST_RESIN"], resin_fair, resin_width, resin_position, resin_limit
            )

        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            result["KELP"] = self.kelp_orders(
                state.order_depths["KELP"], kelp_timespan, kelp_make_width, kelp_take_width, kelp_position, kelp_limit
            )

        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices, 
            "kelp_vwap": self.kelp_vwap, 
            "resin_prices": self.resin_prices,
        })

        conversions = 1
        return result, conversions, traderData