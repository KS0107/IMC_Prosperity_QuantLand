import itertools

currencies = ["Snowballs", "Pizza's", "Silicon Nuggets", "SeaShells"]

exchange_rates = {
    ("Snowballs", "Snowballs"): 1.0,
    ("Snowballs", "Pizza's"): 1.45,
    ("Snowballs", "Silicon Nuggets"): 0.52,
    ("Snowballs", "SeaShells"): 0.72,

    ("Pizza's", "Snowballs"): 0.70,
    ("Pizza's", "Pizza's"): 1.0,
    ("Pizza's", "Silicon Nuggets"): 0.31,
    ("Pizza's", "SeaShells"): 0.48,

    ("Silicon Nuggets", "Snowballs"): 1.95,
    ("Silicon Nuggets", "Pizza's"): 3.10,
    ("Silicon Nuggets", "Silicon Nuggets"): 1.0,
    ("Silicon Nuggets", "SeaShells"): 1.49,

    ("SeaShells", "Snowballs"): 1.34,
    ("SeaShells", "Pizza's"): 1.98,
    ("SeaShells", "Silicon Nuggets"): 0.64,
    ("SeaShells", "SeaShells"): 1.0
}

def brute_force_best_path(initial_amount=500000.0):
    best_amount = 0.0
    best_path = None
    best_details = None
    
    # We start with SeaShells.
    start_currency = "SeaShells"
    
    for path in itertools.product(currencies, repeat=4):
        full_path = list(path) + ["SeaShells"]
        amount = initial_amount
        current_currency = start_currency
        details = []

        for next_currency in full_path:
            rate = exchange_rates[(current_currency, next_currency)]
            before_amount = amount
            amount *= rate
            details.append({
                "from": current_currency,
                "to": next_currency,
                "rate": rate,
                "before": before_amount,
                "after": amount,
                "gain": amount - before_amount,
                "gain_pct": ((amount - before_amount) / before_amount * 100) if before_amount != 0 else 0
            })
            current_currency = next_currency
        
        if amount > best_amount:
            best_amount = amount
            best_path = [start_currency] + full_path
            best_details = details
            
    return best_amount, best_path, best_details

if __name__ == "__main__":
    best_final, best_path, best_details = brute_force_best_path()
    initial_amount = 500000.0
    percentage_gain = ((best_final - initial_amount) / initial_amount) * 100
    
    print("Best conversion path:")
    print(" -> ".join(best_path))
    print(f"\nFinal amount: {best_final:.2f} SeaShells (Percentage gain: {percentage_gain}%)\n")
    
    print("Conversion details per step:")
    for i, step in enumerate(best_details, start=1):
        print(f"\nStep {i}: {step['from']} -> {step['to']}")
        print(f"    Rate: {step['rate']:.4f}")
        print(f"    Amount before: {step['before']:.2f}")
        print(f"    Amount after:  {step['after']:.2f}")
        print(f"    Gain: {step['gain']:.2f} ({step['gain_pct']:.2f}%)")