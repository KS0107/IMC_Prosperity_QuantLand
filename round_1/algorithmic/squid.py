import math
import statistics
from typing import Dict, List
from datamodel import TradingState, OrderDepth, Order

class Trader:
    def __init__(self):
        self.squid_ink_prices = []
        # Parameters for the new model
        self.squid_ink_initial_mean = 2000  # Starting mean at t=0
        self.squid_ink_trend_slope = 6.67e-5  # Linear decline: (2000 - 1800) / 3,000,000
        # High-frequency oscillation
        self.squid_ink_amplitude = 50  # Amplitude of the dominant oscillation
        self.squid_ink_omega = 2 * math.pi / 15000  # Angular frequency (period ≈ 15000)
        self.squid_ink_phase = 0  # Phase shift (to be adjusted if needed)
        self.squid_ink_noise_std = 4  # Standard deviation of noise

    def calculate_volatility(self, prices: List[float], window: int) -> float:
        if len(prices) < 2:
            return 0.0
        window_prices = prices[-min(window, len(prices)):]
        mean = sum(window_prices) / len(window_prices)
        variance = sum((p - mean) ** 2 for p in window_prices) / len(window_prices)
        return (variance ** 0.5) if variance > 0 else 0

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_volume: int, sell_volume: int, fair_value: float) -> tuple[int, int]:
        if position > 0 and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= fair_value:
                quantity = min(position, position_limit + position - sell_volume)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_volume += quantity
        elif position < 0 and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask <= fair_value:
                quantity = min(-position, position_limit - position - buy_volume)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_volume += quantity
        return buy_volume, sell_volume

    def squid_ink_orders(self, state: TradingState, timespan: int, width: float) -> List[Order]:
        orders: List[Order] = []
        buy_volume = 0
        sell_volume = 0

        # Extract order depth for Squid Ink
        order_depth: OrderDepth = state.order_depths.get("SQUID_INK")
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        # Extract position for Squid Ink (default to 0 if not present)
        position = state.position.get("SQUID_INK", 0)
        position_limit = 50  # Assuming the same position limit as before

        # Step 1: Calculate best bid, best ask, and filtered prices
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid

        # Step 2: Calculate mid-price and store it
        mid_price = (mm_ask + mm_bid) / 2
        self.squid_ink_prices.append(mid_price)

        # Step 3: Maintain a fixed-size history based on timespan
        if len(self.squid_ink_prices) > timespan:
            self.squid_ink_prices.pop(0)

        # Step 4: Calculate the fair value using the new model
        # Linear trend
        trend_mean = self.squid_ink_initial_mean - self.squid_ink_trend_slope * state.timestamp
        # High-frequency oscillation
        oscillation = self.squid_ink_amplitude * math.cos(self.squid_ink_omega * state.timestamp + self.squid_ink_phase)
        # Combined fair value
        fair_value = trend_mean + oscillation

        # Step 5: Estimate the trend direction using the derivative
        # Derivative of trend_mean + A * cos(ωt + φ)
        # d/dt(trend_mean) = -slope, d/dt(A * cos(ωt + φ)) = -A * ω * sin(ωt + φ)
        trend_derivative = (
            -self.squid_ink_trend_slope
            - self.squid_ink_amplitude * self.squid_ink_omega * math.sin(self.squid_ink_omega * state.timestamp + self.squid_ink_phase)
        )
        trend = "up" if trend_derivative > 0 else "down" if trend_derivative < 0 else "neutral"

        # Step 6: Volatility Filter (short-term volatility to avoid over-trading)
        short_vol_window = min(5, len(self.squid_ink_prices))
        if short_vol_window > 1:
            short_mean = sum(self.squid_ink_prices[-short_vol_window:]) / short_vol_window
            short_variance = sum((p - short_mean) ** 2 for p in self.squid_ink_prices[-short_vol_window:]) / short_vol_window
            short_std_dev = (short_variance ** 0.5) if short_variance > 0 else 0
        else:
            short_std_dev = 0
        volatility_threshold = 20.0  # Increased due to higher volatility in the graph
        if short_std_dev > volatility_threshold:
            return orders  # Skip trading during high volatility

        # Calculate short-term price range volatility
        price_ranges = []
        for i in range(1, len(self.squid_ink_prices)):
            price_ranges.append(abs(self.squid_ink_prices[i] - self.squid_ink_prices[i - 1]))

        # Adjust window size as needed
        price_range_volatility = statistics.stdev(price_ranges[-5:]) if len(price_ranges) > 5 else 1.0  # Use the last 5 price ranges
  
        # Adjust width based on volatility
        adaptive_width = width * (1 + price_range_volatility / 8)  # Adjust scaling factor as needed

        # Step 7: Mean-Reversion Trading Logic
        # Define buy/sell thresholds based on deviation from the fair value
        deviation = mid_price - fair_value
        threshold = adaptive_width * self.squid_ink_noise_std  # Threshold for mean reversion

        # Adjust thresholds based on trend
        if trend == "up":
            buy_threshold = fair_value - threshold
            sell_threshold = fair_value + threshold + self.squid_ink_noise_std  # Bias toward buying
        elif trend == "down":
            buy_threshold = fair_value - threshold - self.squid_ink_noise_std  # Bias toward selling
            sell_threshold = fair_value + threshold
        else:
            buy_threshold = fair_value - threshold
            sell_threshold = fair_value + threshold

        # Step 8: Stop-Loss Mechanism
        stop_loss_threshold = 400.0  # Increased due to higher volatility
        if position > 0 and mid_price < fair_value - stop_loss_threshold:
            # Close long position at a loss
            quantity = min(position, position_limit + position)
            if quantity > 0:
                orders.append(Order("SQUID_INK", best_bid, -quantity))
                sell_volume += quantity
        elif position < 0 and mid_price > fair_value + stop_loss_threshold:
            # Close short position at a loss
            quantity = min(-position, position_limit - position)
            if quantity > 0:
                orders.append(Order("SQUID_INK", best_ask, quantity))
                buy_volume += quantity

        # Step 9: Mean-Reversion Orders
        if best_ask <= buy_threshold:
            ask_amount = -order_depth.sell_orders[best_ask]
            quantity = min(ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order("SQUID_INK", best_ask, quantity))
                buy_volume += quantity

        if best_bid >= sell_threshold:
            bid_amount = order_depth.buy_orders[best_bid]
            quantity = min(bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order("SQUID_INK", best_bid, -quantity))
                sell_volume += quantity

        # Step 10: Clear any existing positions
        buy_volume, sell_volume = self.clear_position_order(orders, order_depth, position, position_limit, "SQUID_INK", buy_volume, sell_volume, fair_value)

        # Step 11: Market Making Around the Fair Value
        fair_value_rounded = round(fair_value)
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value_rounded + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value_rounded - 1]
        baaf = min(aaf) if aaf else fair_value_rounded + 2
        bbbf = max(bbf) if bbf else fair_value_rounded - 2

        # Calculate order book depth
        order_book_depth = buy_volume + sell_volume

        # Adjust min_spread based on order book depth
        if order_book_depth > 2000:  # Adjust threshold as needed
            adaptive_min_spread = 1  # Narrow spread for deep order book
        elif order_book_depth > 750:
            adaptive_min_spread = 2
        else:
            adaptive_min_spread = 3  # Wider spread for shallow order book

        # Adjust market-making prices based on trend, with wider spread
        # min_spread = 2  # Ensure at least 2 points of profit per trade
        if trend == "up":
            buy_price = int(bbbf)  # Wider spread
            sell_price = int(baaf) + 1  # Skew sell price higher
        elif trend == "down":
            buy_price = int(bbbf) - 1  # Skew buy price lower
            sell_price = int(baaf)  # Wider spread
        else:
            buy_price = int(bbbf)
            sell_price = int(baaf)

        # Skip market-making if spread is too small
        if sell_price - buy_price < adaptive_min_spread:
            return orders

        # Use volatility to adjust order size
        squid_ink_volatility = self.calculate_volatility(self.squid_ink_prices, window=15)
        max_order_size = max(5, int(20 / (1 + squid_ink_volatility)))

        # Place buy order
        buy_quantity = position_limit - (position + buy_volume)
        if buy_quantity > 0:
            order_size = min(buy_quantity, max_order_size)
            orders.append(Order("SQUID_INK", buy_price, order_size))

        # Place sell order
        sell_quantity = position_limit + (position - sell_volume)
        if sell_quantity > 0:
            order_size = min(sell_quantity, max_order_size)
            orders.append(Order("SQUID_INK", sell_price, -order_size))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Main method called by the trading engine.
        """
        result = {}
        
        # Parameters for the strategy
        timespan = 10  # Lookback period for historical prices
        width = 30.0   # Width for mean-reversion thresholds

        # Generate orders for Squid Ink
        squid_ink_orders = self.squid_ink_orders(state, timespan, width)
        result["SQUID_INK"] = squid_ink_orders

        # Add strategies for other products (Kelp, Rainforest Resin) here if needed
        for product in state.order_depths.keys():
            if product != "SQUID_INK":
                result[product] = []

        traderData = ""
        conversions = 0
        return result, conversions, traderData