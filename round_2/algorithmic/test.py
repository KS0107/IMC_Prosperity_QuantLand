from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBE = "DJEMBE"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "kelp_min_edge": 2,
    },
    Product.SPREAD: {
        "default_spread_mean": 379.50439988484239,  # Adjust for PICNIC_BASKET1 if needed
        "default_spread_std": 76.07966,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 58,
    },
    # NEW: Parameters for PICNIC_BASKET2 spread
    Product.PICNIC_BASKET2: {
        "default_spread_mean": 0,  # Estimate from data or set dynamically
        "default_spread_std": 50,  # Adjust based on observed volatility
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 80,  # Smaller than limit (100), adjust as needed
    },
}


# NEW: Define weights for both baskets
BASKET_WEIGHTS = {
    Product.PICNIC_BASKET1: {
        Product.JAMS: 3,       # Corrected from 4 to 3
        Product.CROISSANTS: 6,
        Product.DJEMBE: 1,
    },
    Product.PICNIC_BASKET2: {
        Product.JAMS: 2,
        Product.CROISSANTS: 4,
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 20,
            Product.KELP: 20,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.JAMS: 350,
            Product.CROISSANTS: 250,
            Product.DJEMBE: 60,
        }

    # [Unchanged methods: take_best_orders, take_best_orders_with_adverse, market_make, 
    # clear_position_order, kelp_fair_value, make_resin_orders, take_orders, 
    # clear_orders, make_kelp_orders, orchids_implied_bid_ask, orchids_arb_take, 
    # orchids_arb_clear, orchids_arb_make]
    # Skipping these for brevity; assume they remain as provided

    def get_swmid(self, order_depth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth], basket_product: str
    ) -> OrderDepth:
        # Select weights based on basket
        weights = BASKET_WEIGHTS[basket_product]
        components = weights.keys()

        # Initialize synthetic order depth
        synthetic_order_price = OrderDepth()

        # Calculate best bid and ask for each component
        component_bids = {}
        component_asks = {}
        for component in components:
            component_bids[component] = (
                max(order_depths[component].buy_orders.keys())
                if order_depths[component].buy_orders
                else 0
            )
            component_asks[component] = (
                min(order_depths[component].sell_orders.keys())
                if order_depths[component].sell_orders
                else float("inf")
            )

        # Calculate implied bid and ask for synthetic basket
        implied_bid = sum(
            component_bids[comp] * weights[comp] for comp in components
        )
        implied_ask = sum(
            component_asks[comp] * weights[comp] for comp in components
        )

        # Calculate volumes
        if implied_bid > 0:
            bid_volumes = [
                order_depths[comp].buy_orders.get(component_bids[comp], 0) // weights[comp]
                for comp in components
            ]
            implied_bid_volume = min(bid_volumes)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            ask_volumes = [
                -order_depths[comp].sell_orders.get(component_asks[comp], 0) // weights[comp]
                for comp in components
            ]
            implied_ask_volume = min(ask_volumes)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth], basket_product: str
    ) -> Dict[str, List[Order]]:
        component_orders = {comp: [] for comp in BASKET_WEIGHTS[basket_product].keys()}
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)
        
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity

            # Determine component prices
            component_prices = {}
            for comp in BASKET_WEIGHTS[basket_product].keys():
                if quantity > 0 and price >= best_ask:
                    component_prices[comp] = min(order_depths[comp].sell_orders.keys())
                elif quantity < 0 and price <= best_bid:
                    component_prices[comp] = max(order_depths[comp].buy_orders.keys())
                else:
                    continue

            # Create component orders
            for comp, weight in BASKET_WEIGHTS[basket_product].items():
                if comp in component_prices:
                    comp_order = Order(comp, component_prices[comp], quantity * weight)
                    component_orders[comp].append(comp_order)

        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
        basket_product: str,
    ):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[basket_product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_basket_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_basket_order_depth.buy_orders[synthetic_bid_price])

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(basket_product, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, basket_product
            )
            aggregate_orders[basket_product] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_basket_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_basket_order_depth.sell_orders[synthetic_ask_price])

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(basket_product, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, basket_product
            )
            aggregate_orders[basket_product] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        basket_product: str,
        basket_position: int,
        spread_data: Dict[str, Any],
        spread_params: Dict[str, Any],
    ):
        if basket_product not in order_depths:
            return None

        basket_order_depth = order_depths[basket_product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if len(spread_data["spread_history"]) < spread_params["spread_std_window"]:
            return None
        elif len(spread_data["spread_history"]) > spread_params["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        zscore = (spread - spread_params["default_spread_mean"]) / spread_std

        if zscore >= spread_params["zscore_threshold"]:
            if basket_position != -spread_params["target_position"]:
                return self.execute_spread_orders(
                    -spread_params["target_position"], basket_position, order_depths, basket_product
                )
        elif zscore <= -spread_params["zscore_threshold"]:
            if basket_position != spread_params["target_position"]:
                return self.execute_spread_orders(
                    spread_params["target_position"], basket_position, order_depths, basket_product
                )

        spread_data["prev_zscore"] = zscore
        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        # Initialize traderObject for spreads
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        if Product.PICNIC_BASKET2 not in traderObject:
            traderObject[Product.PICNIC_BASKET2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        # RAINFOREST_RESIN logic (unchanged)
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                resin_position,
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            resin_make_orders, _, _ = self.make_resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["volume_limit"],
            )
            result[Product.RAINFOREST_RESIN] = resin_take_orders + resin_clear_orders + resin_make_orders

        # KELP logic (unchanged)
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject)
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["take_width"],
                kelp_position,
                self.params[Product.KELP]["prevent_adverse"],
                self.params[Product.KELP]["adverse_volume"],
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["clear_width"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            kelp_make_orders, _, _ = self.make_kelp_orders(
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["kelp_min_edge"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        # Spread trading for both baskets
        basket1_position = state.position.get(Product.PICNIC_BASKET1, 0)
        spread1_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket1_position,
            traderObject[Product.SPREAD],
            self.params[Product.SPREAD],
        )

        basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)
        spread2_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket2_position,
            traderObject[Product.PICNIC_BASKET2],
            self.params[Product.PICNIC_BASKET2],
        )

        # Combine orders, checking position limits
        for product in [
            Product.PICNIC_BASKET1,
            Product.PICNIC_BASKET2,
            Product.JAMS,
            Product.CROISSANTS,
            Product.DJEMBE,
        ]:
            result[product] = []
            total_quantity = state.position.get(product, 0)

            if spread1_orders and product in spread1_orders:
                for order in spread1_orders[product]:
                    new_total = total_quantity + order.quantity
                    if abs(new_total) <= self.LIMIT[product]:
                        result[product].append(order)
                        total_quantity = new_total

            if spread2_orders and product in spread2_orders:
                for order in spread2_orders[product]:
                    new_total = total_quantity + order.quantity
                    if abs(new_total) <= self.LIMIT[product]:
                        result[product].append(order)
                        total_quantity = new_total

        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData