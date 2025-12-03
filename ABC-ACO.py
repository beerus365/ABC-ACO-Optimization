import numpy as np
import matplotlib.pyplot as plt
import random
import time


# -------------------------
# Core Utility Functions
# -------------------------
def create_node(num_node=100):
    """Create a set of node with random coordinates and labels."""
    node = np.random.uniform(0, 100, size=(num_node, 2))
    labels = [f'N {i + 1}' for i in range(num_node)]
    return node, labels


def compute_distance_matrix(node, peak_hour=False, event_traffic=False):
    """Compute the pairwise distance matrix considering traffic conditions."""
    n = len(node)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            base_distance = np.linalg.norm(node[i] - node[j])
            traffic_factor = 1.0

            if peak_hour:
                traffic_factor += 0.5
            if event_traffic and random.random() < 0.2:
                traffic_factor += 0.3
            if random.random() < 0.1:
                traffic_factor += 0.2

            dist_matrix[i, j] = base_distance * traffic_factor

    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix


def compute_path_length(path, dist_matrix):
    return sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))


# -------------------------
# Visualization
# -------------------------

NODE_COLOR_FINAL = '#006400'
NODE_SIZE_FINAL = 60


def visualize_single_route(node, labels, best_path=None, path_length=None, title="Route Visualization"):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#F9F9F9')
    ax.tick_params(axis='both', which='both', length=0)

    for i, label in enumerate(labels):
        plt.text(node[i, 0], node[i, 1], label, fontsize=8, ha='right')

    # Draw the Best Path (Route)
    if best_path is not None:
        path_x = node[best_path, 0]
        path_y = node[best_path, 1]

        ax.plot(path_x, path_y,
                color='#006B9A',
                linewidth=2.5,
                alpha=1.0,
                linestyle='-',
                marker='o', markersize=6, markerfacecolor='none', markeredgecolor='#006B9A', markeredgewidth=1.5,
                zorder=1, label='Optimal Route')

        # Highlight Start/End Point (Depot)
        start_node_index = best_path[0]
        ax.scatter(node[start_node_index, 0], node[start_node_index, 1],
                   color='#FF3B30', s=250, zorder=4, marker='*', edgecolor='white', linewidth=2,
                   label='Depot (Start/End)')

    # Draw all node (Nodes) - Layered on top
    ax.scatter(node[:, 0], node[:, 1],
               color=NODE_COLOR_FINAL,
               s=NODE_SIZE_FINAL,
               alpha=1.0,
               edgecolor=NODE_COLOR_FINAL,
               linewidth=0.5,
               zorder=3, label='Intersection')

    # Final Touches
    full_title = f"{title}\nBest Length: {path_length:.2f}" if path_length is not None else title
    ax.set_title(full_title, fontsize=18, fontweight='bold', color='#333333')
    ax.set_xlabel("X Coordinate (Grid)", color='#333333')
    ax.set_ylabel("Y Coordinate (Grid)", color='#333333')
    ax.grid(True, linestyle=':', alpha=0.3, color='#CCCCCC')
    ax.legend(loc='lower right', frameon=False, fontsize=10, labelcolor='#333333')
    ax.tick_params(axis='x', colors='#333333')
    ax.tick_params(axis='y', colors='#333333')
    plt.tight_layout()
    plt.show()  # Pop-up display


def visualize_comparison_side_by_side(nodes, labels, path1, len1, title1, path2, len2, title2):
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Common styling for both subplots
    for ax in [ax1, ax2]:
        ax.set_facecolor('#F9F9F9')
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xlabel("X Coordinate (Grid)", color='#333333')
        ax.set_ylabel("Y Coordinate (Grid)", color='#333333')
        ax.grid(True, linestyle=':', alpha=0.3, color='#CCCCCC')
        ax.tick_params(axis='x', colors='#333333')
        ax.tick_params(axis='y', colors='#333333')

        # Draw Paths first
        path = path1 if ax == ax1 else path2
        route_len = len1 if ax == ax1 else len2

        if path is not None:
            path_x = nodes[path, 0]
            path_y = nodes[path, 1]
            ax.plot(path_x, path_y,
                    color='#006B9A', linewidth=3.0, alpha=1.0, linestyle='-',
                    marker='o', markersize=6, markerfacecolor='none', markeredgecolor='#006B9A', markeredgewidth=1.5,
                    zorder=1, label='Optimal Route')
            start_node_index = path[0]
            ax.scatter(nodes[start_node_index, 0], nodes[start_node_index, 1],
                       color='#FF3B30', s=250, zorder=4, marker='*', edgecolor='white', linewidth=2, label='Depot')

        # Draw node (Nodes) on top
        ax.scatter(nodes[:, 0], nodes[:, 1],
                   color=NODE_COLOR_FINAL,
                   s=NODE_SIZE_FINAL,
                   alpha=1.0,
                   edgecolor=NODE_COLOR_FINAL,
                   linewidth=0.5,
                   zorder=3, label='Intersection')

    # Set titles and legends for each subplot
    ax1.set_title(f"{title1}\nBest Length: {len1:.2f}", fontsize=18, fontweight='bold', color='#333333')
    ax1.legend(loc='lower right', frameon=False, fontsize=10, labelcolor='#333333')

    ax2.set_title(f"{title2}\nBest Length: {len2:.2f}", fontsize=18, fontweight='bold', color='#333333')
    ax2.legend(loc='lower right', frameon=False, fontsize=10, labelcolor='#333333')

    plt.tight_layout()
    plt.show()  # Pop-up display


# -------------------------
# Optimization Logic (The Engines)
# -------------------------
def local_search_two_opt_fast(path, dist_matrix):
    """Fast First-Improvement 2-Opt."""
    current_path = path.copy()
    n = len(current_path) - 1

    for i in range(1, n - 1):
        for k in range(i + 1, n):
            u1, v1 = current_path[i - 1], current_path[i]
            u2, v2 = current_path[k], current_path[k + 1]

            if dist_matrix[u1, u2] + dist_matrix[v1, v2] < dist_matrix[u1, v1] + dist_matrix[u2, v2]:
                new_path = current_path[:i] + current_path[i:k + 1][::-1] + current_path[k + 1:]
                return new_path, compute_path_length(new_path, dist_matrix)

    return current_path, compute_path_length(current_path, dist_matrix)


def apply_abc(elite_solutions, dist_matrix, onlooker_count=10, limit=10):
    """ABC Refinement logic."""
    population = [list(sol[0]) for sol in elite_solutions]
    fitness = [sol[1] for sol in elite_solutions]
    trials = [0] * len(population)

    # 1. Employed Bees
    for i in range(len(population)):
        cand, cand_len = local_search_two_opt_fast(population[i], dist_matrix)
        if cand_len < fitness[i]:
            population[i], fitness[i], trials[i] = cand, cand_len, 0
        else:
            trials[i] += 1

    # 2. Onlooker Bees
    inv_fit = [1.0 / (f + 1e-9) for f in fitness]
    total_inv = sum(inv_fit)
    probs = [x / total_inv for x in inv_fit] if total_inv > 0 else [1 / len(fitness)] * len(fitness)

    for _ in range(onlooker_count):
        idx = random.choices(range(len(population)), probs)[0]
        cand, cand_len = local_search_two_opt_fast(population[idx], dist_matrix)
        if cand_len < fitness[idx]:
            population[idx], fitness[idx], trials[idx] = cand, cand_len, 0

    # 3. Scout Bees
    for i in range(len(population)):
        if trials[i] >= limit:
            # Reset this bee with a random path (simple shuffle)
            new_path = list(range(dist_matrix.shape[0]))
            random.shuffle(new_path)
            new_path.append(new_path[0])
            population[i] = new_path
            fitness[i] = compute_path_length(new_path, dist_matrix)
            trials[i] = 0

    return [(population[i], fitness[i]) for i in range(len(population))]


def ant_colony_optimization_engine(node, dist_matrix, n_ants, n_iterations, Q, rho, alpha, beta, elite_k):
    """Generic ACO Engine."""
    n = len(node)
    pheromone_matrix = np.ones_like(dist_matrix) + 1e-6
    best_path, best_path_length = None, float('inf')

    for iteration in range(n_iterations):
        ants_paths = []
        for _ in range(n_ants):
            path = [random.randint(0, n - 1)]
            visited = {path[0]}
            current = path[0]

            for _ in range(n - 1):
                unvisited = [j for j in range(n) if j not in visited]
                if not unvisited: break

                # Vectorized-like probability calculation
                probs = []
                for j in unvisited:
                    p = (pheromone_matrix[current, j] ** alpha) * ((1.0 / (dist_matrix[current, j] + 1e-9)) ** beta)
                    probs.append(p)

                total = sum(probs)
                if total <= 0:
                    next_node = random.choice(unvisited)
                else:
                    next_node = random.choices(unvisited, [p / total for p in probs])[0]

                path.append(next_node)
                visited.add(next_node)
                current = next_node

            path.append(path[0])
            ants_paths.append((path, compute_path_length(path, dist_matrix)))

        # Update Best
        iter_best = min(ants_paths, key=lambda x: x[1])
        if iter_best[1] < best_path_length:
            best_path, best_path_length = iter_best[0][:], iter_best[1]

        # Hybrid Refinement & Pheromone Update
        ants_paths.sort(key=lambda x: x[1])
        if elite_k > 0:
            refined = apply_abc(ants_paths[:elite_k], dist_matrix)
            ants_paths[:elite_k] = refined
            # Update global best again if refinement found something better
            if refined[0][1] < best_path_length:
                best_path, best_path_length = refined[0][0][:], refined[0][1]

        pheromone_matrix *= (1 - rho)
        for path, length in ants_paths[:max(len(ants_paths) // 2, elite_k)]:  # Deposit for top 50% or elites
            deposit = Q / (length + 1e-9)
            for i in range(len(path) - 1):
                pheromone_matrix[path[i], path[i + 1]] += deposit
                pheromone_matrix[path[i + 1], path[i]] += deposit

        print(f"Iteration {iteration + 1}: Best Path Length = {best_path_length:.4f}")
    return best_path, best_path_length


# -------------------------
# Configuration Wrappers (The Interface)
# -------------------------
def run_baseline_aco(node, dist_matrix):
    """Runs Standard ACO with predefined settings."""
    # Configuration
    params = {
        'n_ants': 30,
        'n_iterations': 50,
        'Q': 100,
        'rho': 0.2,
        'alpha': 1.0,
        'beta': 2.0,
        'elite_k': 0  # Disable Hybrid
    }

    start = time.time()
    path, length = ant_colony_optimization_engine(node, dist_matrix, **params)
    duration = time.time() - start
    return path, length, duration


def run_hybrid_aco(node, dist_matrix):
    """Runs Hybrid ACO+ABC with tuned settings."""
    # Configuration
    params = {
        'n_ants': 30,
        'n_iterations': 50,
        'Q': 100,
        'rho': 0.2,
        'alpha': 1.0,
        'beta': 2.0,
        'elite_k': 10  # Enable Hybrid (Refine top 5)
    }

    start = time.time()
    path, length = ant_colony_optimization_engine(node, dist_matrix, **params)
    duration = time.time() - start
    return path, length, duration


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Global Seed based on time (Changes every run)
    SEED = 48

    # ==========================================
    # TEST 1: 100 NODES (Fast Check)
    # ==========================================
    N_NODES_1 = 100

    print(f"🚀 Running Comparison: {N_NODES_1} node (Seed: {SEED})\n")

    # --- 1A. Baseline (100 Nodes) ---
    print("1. Running Baseline ACO...")
    random.seed(SEED)
    np.random.seed(SEED)
    nodes, labels = create_node(N_NODES_1)
    dists = compute_distance_matrix(nodes, peak_hour=True)

    path_base_1, len_base_1, time_base_1 = run_baseline_aco(nodes, dists)

    # --- 1B. Hybrid (100 Nodes - Strict Reset) ---
    print("\n2. Running Hybrid ACO+ABC...")
    random.seed(SEED)
    np.random.seed(SEED)
    nodes, labels = create_node(N_NODES_1)
    dists = compute_distance_matrix(nodes, peak_hour=True)

    path_hyb_1, len_hyb_1, time_hyb_1 = run_hybrid_aco(nodes, dists)

    # --- Results 1 ---
    print("\n" + "=" * 45)
    print(f"          RESULTS: {N_NODES_1} node")
    print("-" * 45)
    print(f"{'Algorithm':<20} | {'Length':<10} | {'Time (s)':<10}")
    print("-" * 45)
    print(f"{'ACO Only':<20} | {len_base_1:<10.4f} | {time_base_1:<10.2f}")
    print(f"{'Hybrid ACO+ABC':<20} | {len_hyb_1:<10.4f} | {time_hyb_1:<10.2f}")
    print("=" * 45 + "\n")

    # Visuals for 100 Nodes
    visualize_single_route(nodes, labels, path_base_1, len_base_1, f"Baseline ACO ({N_NODES_1} Nodes)")
    visualize_single_route(nodes, labels, path_hyb_1, len_hyb_1, f"Hybrid ACO ({N_NODES_1} Nodes)")
    visualize_comparison_side_by_side(nodes, labels, path_base_1, len_base_1, "Baseline",
                                      path_hyb_1, len_hyb_1, "Hybrid")

    # # ==========================================
    # # TEST 2: 500 NODES (Stress Test)
    # # ==========================================
    # N_NODES_2 = 300

    # print(f"\n\n🚀 Running Comparison: {N_NODES_2} node (Seed: {SEED})")

    # # --- 2A. Baseline (300 Nodes) ---
    # print("1. Running Baseline ACO...")
    # random.seed(SEED)
    # np.random.seed(SEED)
    # nodes_500, labels_500 = create_node(N_NODES_2)
    # dists_500 = compute_distance_matrix(nodes_500, peak_hour=True)

    # path_base_2, len_base_2, time_base_2 = run_baseline_aco(nodes_500, dists_500)

    # # --- 2B. Hybrid (300 Nodes - Strict Reset) ---
    # print("2. Running Hybrid ACO+ABC...")
    # random.seed(SEED)
    # np.random.seed(SEED)
    # nodes_500, labels_500 = create_node(N_NODES_2)
    # dists_500 = compute_distance_matrix(nodes_500, peak_hour=True)

    # path_hyb_2, len_hyb_2, time_hyb_2 = run_hybrid_aco(nodes_500, dists_500)

    # # --- Results 2 ---
    # print("\n" + "="*45)
    # print(f"RESULTS: {N_NODES_2} node")
    # print("-" * 45)
    # print(f"{'Algorithm':<20} | {'Length':<10} | {'Time (s)':<10}")
    # print("-" * 45)
    # print(f"{'ACO Only':<20} | {len_base_2:<10.4f} | {time_base_2:<10.2f}")
    # print(f"{'Hybrid ACO+ABC':<20} | {len_hyb_2:<10.4f} | {time_hyb_2:<10.2f}")
    # print("="*45 + "\n")

    # # Visuals for 500 Nodes (Side-by-Side is best for large maps)
    # visualize_comparison_side_by_side(nodes_500, labels_500,
    #                                   path_base_2, len_base_2, f"Baseline ({N_NODES_2})",
    #                                   path_hyb_2, len_hyb_2, f"Hybrid ({N_NODES_2})")