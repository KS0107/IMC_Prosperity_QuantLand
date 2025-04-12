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
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"


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
    Product.SPREAD1: {
        "default_spread_mean": 26.60,  # Adjust for PICNIC_BASKET1 if needed
        "default_spread_std": 27.05,
        "spread_std_window": 45,
        "zscore_threshold": 3,
        "target_position": 58,
    },
    # NEW: Parameters for PICNIC_BASKET2 spread
    Product.SPREAD2: {
        "default_spread_mean": 105.38,  # Estimate from data or set dynamically
        "default_spread_std": 27.12,  # Adjust based on observed volatility
        "spread_std_window": 45,
        "zscore_threshold": 3,
        "target_position": 70,  # Smaller than limit (100), adjust as needed
    },
}

BASKET_WEIGHTS = {
    Product.PICNIC_BASKET1: {
        Product.JAMS: 3,       
        Product.CROISSANTS: 6,
        Product.DJEMBES: 1,
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
            Product.DJEMBES: 60,
        }

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def make_resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        baaf = min(
            [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]
        )
        bbbf = max(
            [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        )

        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3  # still want edge 2 if position is not a concern

        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3  # still want edge 2 if position is not a concern

        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_kelp_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price
            for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price
            for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

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

        # Check if all components are present in order_depths
        missing_components = [comp for comp in components if comp not in order_depths]
        if missing_components:
            print(f"Missing components for {basket_product}: {missing_components}")
            return OrderDepth()

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
            if not order_depths[component].buy_orders:
                print(f"No buy orders for {component} in {basket_product}")
            if not order_depths[component].sell_orders:
                print(f"No sell orders for {component} in {basket_product}")

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
           
        weights = BASKET_WEIGHTS[basket_product]
        components = weights.keys()

        component_orders = {comp: [] for comp in components}

        # Early return if no synthetic orders
        if not synthetic_orders:
            return component_orders

        # Check if all components have order depths
        missing_components = [comp for comp in components if comp not in order_depths]
        if missing_components:
            print(f"Missing components in convert_synthetic for {basket_product}: {missing_components}")
            return component_orders

        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)
        if not synthetic_order_depth.buy_orders and not synthetic_order_depth.sell_orders:
            print(f"No valid synthetic order depth for {basket_product}")
            return component_orders
        
        best_bid = (
            max(synthetic_order_depth.buy_orders.keys())
            if synthetic_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_order_depth.sell_orders.keys())
            if synthetic_order_depth.sell_orders
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

        if basket_product not in order_depths:
            print(f"No order depth for {basket_product}")
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[basket_product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)

        if not synthetic_order_depth.buy_orders and not synthetic_order_depth.sell_orders:
            print(f"No valid synthetic order depth for {basket_product} in execute_spread_orders")
            return None

        if target_position > basket_position:
            if not basket_order_depth.sell_orders or not synthetic_order_depth.buy_orders:
                print(f"Missing sell orders for {basket_product} or buy orders for synthetic")
                return None

            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            if execute_volume == 0:
                return None

            basket_orders = [Order(basket_product, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, basket_product
            )
            aggregate_orders[basket_product] = basket_orders
            return aggregate_orders

        else:
            if not basket_order_depth.buy_orders or not synthetic_order_depth.sell_orders:
                print(f"Missing buy orders for {basket_product} or sell orders for synthetic")
                return None

            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            if execute_volume == 0:
                return None

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

        # Check if all components are present
        weights = BASKET_WEIGHTS[basket_product]
        missing_components = [comp for comp in weights.keys() if comp not in order_depths]
        if missing_components:
            print(f"Missing components for {basket_product} in spread_orders: {missing_components}")
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
        if spread_std == 0:
            print(f"Zero spread std for {basket_product}")
            return None
        
        
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
        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
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
            traderObject[Product.SPREAD1],
            self.params[Product.SPREAD1],
        )

        basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)
        spread2_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket2_position,
            traderObject[Product.SPREAD2],
            self.params[Product.SPREAD2],
        )

        # Combine orders, checking position limits
        for product in [
            Product.PICNIC_BASKET1,
            Product.PICNIC_BASKET2,
            Product.JAMS,
            Product.CROISSANTS,
            Product.DJEMBES,
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