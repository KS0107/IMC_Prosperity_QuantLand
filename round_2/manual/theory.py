import math

# ----------------------------------------------------------------
# 1) Hardcode containers: each is (multiplier, base_inhabitants)
# ----------------------------------------------------------------
containers = [
    (10, 1),
    (17, 1),
    (20, 2),
    (31, 2),
    (37, 3),
    (50, 4),
    (73, 4),
    (80, 6),
    (89, 8),
    (90, 10)
]

N_TEAMS = 1000  # Number of identical teams

# ----------------------------------------------------------------
# 2) Possible actions = "NONE", "SINGLE(i)", "DOUBLE(i,j)"
# ----------------------------------------------------------------
ACTIONS = []
ACTIONS.append(("NONE",))

CONTAINER_COUNT = len(containers)

# Single picks
for i in range(CONTAINER_COUNT):
    ACTIONS.append(("SINGLE", i))

# Double picks
for i in range(CONTAINER_COUNT):
    for j in range(i+1, CONTAINER_COUNT):
        ACTIONS.append(("DOUBLE", i, j))

# Probability distribution over these actions
dist_size = len(ACTIONS)
dist = [1.0 / dist_size] * dist_size  # uniform start

def compute_occupants_and_total_picks(dist):
    """
    Returns:
      occupant[i]: occupant count for container i
      total_picks: sum of picks across all containers
                   (each SINGLE is +1 pick, each DOUBLE is +2)
    occupant[i] = base_inhabitants_i + (sum_of_teams_that_pick_i)
    sum_of_teams_that_pick_i = dist[a]*N_TEAMS for all a that pick i.
    total_picks = sum( occupant[i] - base_inhabitants[i] ), or just
                  sum_{action} dist[action]*N_TEAMS*(1 if SINGLE, 2 if DOUBLE).
    """
    occupant = [c[1] for c in containers]  # start from base_inhabitants
    total_picks = 0.0

    for idx, action in enumerate(ACTIONS):
        fraction = dist[idx]
        teams_for_action = fraction * N_TEAMS
        if action[0] == "NONE":
            # no containers
            continue
        elif action[0] == "SINGLE":
            i = action[1]
            occupant[i] += teams_for_action
            total_picks += teams_for_action  # +1 pick per team
        elif action[0] == "DOUBLE":
            i, j = action[1], action[2]
            occupant[i] += teams_for_action
            occupant[j] += teams_for_action
            total_picks += 2*teams_for_action  # +2 picks per team
    return occupant, total_picks

def payoff_single_container(container_index, occupant_i, total_picks):
    """
    Payoff for picking container i alone, under the new rule:
      base_value = multiplier*10000
      denominator = occupant_i + (percentage_of_picks_i * 100)
    where percentage_of_picks_i = occupant_i / total_picks ( occupant_i includes your presence ).

    Return (base_value / denominator).
    """
    multiplier, base_inhabitants = containers[container_index]
    base_value = multiplier*10000.0
    if total_picks < 1e-9:
        # no picks => denominator can't be computed => return 0
        return 0.0

    # occupant_i includes your presence in single-pop approximation
    # fraction_of_picks = occupant_i / total_picks
    # added_amount = fraction_of_picks * 100
    frac = occupant_i / total_picks
    denominator = occupant_i + 100.0*frac  # occupant + occupant_i/total_picks*100

    if denominator < 1e-9:
        return 0.0

    return base_value / denominator

def payoff_of_action(action, occupant, total_picks):
    """
    If "NONE": payoff=0
    If "SINGLE"(i):
      payoff = payoff_single_container(i) 
    If "DOUBLE"(i,j):
      payoff = payoff_single_container(i) + payoff_single_container(j) - 50000
    """
    if action[0] == "NONE":
        return 0.0

    elif action[0] == "SINGLE":
        i = action[1]
        return payoff_single_container(i, occupant[i], total_picks)

    elif action[0] == "DOUBLE":
        i, j = action[1], action[2]
        payoff_i = payoff_single_container(i, occupant[i], total_picks)
        payoff_j = payoff_single_container(j, occupant[j], total_picks)
        return payoff_i + payoff_j - 50000.0

    return 0.0

def update_distribution(dist, steps=2000, lr=0.1, temperature=50000.0):
    """
    We do fictitious play/softmax updates. For each iteration:
      1) occupant, total_picks = compute_occupants_and_total_picks(dist)
      2) payoff[a] = payoff_of_action(a, occupant, total_picks)
      3) new_dist = softmax(payoff[a] / temperature)
      4) dist = (1-lr)*dist + lr*new_dist
    Return final distribution
    """
    for step in range(steps):
        occupant, total_picks = compute_occupants_and_total_picks(dist)
        payoffs = [payoff_of_action(a, occupant, total_picks) for a in ACTIONS]

        # Softmax
        exps = [math.exp(p / temperature) for p in payoffs]
        sum_exps = sum(exps)
        new_dist = [e/sum_exps for e in exps]

        # blend
        dist = [(1-lr)*dist[i] + lr*new_dist[i] for i in range(len(dist))]

    return dist

def main():
    global dist
    # run
    dist = update_distribution(dist, steps=400, lr=0.2, temperature=50000.0)
    
    occupant, total_picks = compute_occupants_and_total_picks(dist)
    payoffs = [payoff_of_action(a, occupant, total_picks) for a in ACTIONS]

    # Print final distribution for actions >= 1%
    print("=== Approx. Mixed Strategy Equilibrium ===")
    for i, a in enumerate(ACTIONS):
        if dist[i] > 0.01:
            print(f"Action {a}, p={dist[i]:.3f}, payoff={payoffs[i]:.2f}")

    print("\n--- Container occupant + fraction-of-picks ---")
    for i, (mult, base) in enumerate(containers):
        frac = occupant[i]/total_picks*100 if total_picks>1e-9 else 0
        print(f"Container {i}: occupant={occupant[i]:.2f}, fraction_of_picks={frac:.1f}%, multiplier={mult}")

    # Best single deviant action
    best_idx = max(range(len(ACTIONS)), key=lambda idx: payoffs[idx])
    print(f"\nBest unilateral deviant action: {ACTIONS[best_idx]}, payoff={payoffs[best_idx]:.2f}")

if __name__ == "__main__":
    main()