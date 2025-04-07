def calculate_exchange(amount, exchange_rate):
    """Calculates the exchanged amount based on the given rate."""
    return amount * exchange_rate

def find_best_trade(exchange_rates, capital, start_currency, end_currency, max_trades=5, min_trades=2):
    """Finds the best trade sequence to maximize the end currency with at least min_trades."""
    currencies = list(exchange_rates.keys())
    best_result = 0
    best_path = []
    all_paths = []  # To track all paths
    all_profitable_paths = []  # To track all profitable paths

    def find_path(current_path, current_amount, remaining_trades):
        nonlocal best_result, best_path, all_profitable_paths

        if (current_path[-1] == end_currency and 
            len(current_path) >= min_trades + 1):  # Ensure at least min_trades
            all_paths.append((current_path[1:-1], current_amount))  # Store the path and amount
            if current_amount > capital:  # Only log profitable paths
                all_profitable_paths.append((current_path[:], current_amount))
            if current_amount > best_result:
                best_result = current_amount
                best_path = current_path[:]

        if remaining_trades == 0:
            return

        last_currency = current_path[-1]
        for next_currency in currencies:
            if last_currency != next_currency:  # Avoid self-trades
                exchange_rate = exchange_rates[last_currency][next_currency]
                new_amount = calculate_exchange(current_amount, exchange_rate)
                find_path(current_path + [next_currency], new_amount, remaining_trades - 1)

    find_path([start_currency], capital, max_trades)

    # Print all profitable paths for inspection
    for path, amount in sorted(all_profitable_paths, key=lambda x: x[1], reverse=True):
        print(f"Path: {path}, Amount: {amount}, Profit: {amount - capital}")

    print(f"\nTotal Profitable Paths Found: {len(all_profitable_paths)}")
    
    return best_result, best_path

# Exchange rates
exchange_rates = {
    "SeaShells": {"Snowballs": 1.34, "Pizzas": 1.98, "Silicon Nuggets": 0.64, "SeaShells": 1},
    "Silicon Nuggets": {"Snowballs": 1.95, "Pizzas": 3.1, "Silicon Nuggets": 1, "SeaShells": 1.49},
    "Pizzas": {"Snowballs": 0.7, "Pizzas": 1, "Silicon Nuggets": 0.31, "SeaShells": 0.48},
    "Snowballs": {"Snowballs": 1, "Pizzas": 1.45, "Silicon Nuggets": 0.52, "SeaShells": 0.72},
}

capital = 2000000
start_currency = "SeaShells"
end_currency = "SeaShells"
max_trades = 5
min_trades = 1

best_result, best_path = find_best_trade(exchange_rates, capital, start_currency, end_currency, max_trades, min_trades)

print(f"\nBest Trade Path: {best_path}")
print(f"Resulting Amount: {best_result}")
print(f"Percentage Gain: {((best_result - capital) / capital) * 100:.3f}%")