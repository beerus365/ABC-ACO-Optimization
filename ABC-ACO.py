import numpy as np
import matplotlib.pyplot as plt
import random
import time


# -------------------------
# Core Utility Functions
# -------------------------
def create_intersections(num_intersections=100):
    """Create a set of intersections with random coordinates and labels."""
    intersections = np.random.uniform(0, 100, size=(num_intersections, 2))
    labels = [f'Intersection {i + 1}' for i in range(num_intersections)]
    return intersections, labels


def compute_distance_matrix(intersections, peak_hour=False, event_traffic=False):
    """Compute the pairwise distance matrix considering traffic conditions."""
    n = len(intersections)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            base_distance = np.linalg.norm(intersections[i] - intersections[j])
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
            new_path = list(range(dist_matrix.shape[0]))
            random.shuffle(new_path)
            new_path.append(new_path[0])
            population[i] = new_path
            fitness[i] = compute_path_length(new_path, dist_matrix)
            trials[i] = 0

    return [(population[i], fitness[i]) for i in range(len(population))]


def ant_colony_optimization_engine(intersections, dist_matrix, n_ants, n_iterations, Q, rho, alpha, beta, elite_k):
    """Generic ACO Engine."""
    n = len(intersections)
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

        iter_best = min(ants_paths, key=lambda x: x[1])
        if iter_best[1] < best_path_length:
            best_path, best_path_length = iter_best[0][:], iter_best[1]

        ants_paths.sort(key=lambda x: x[1])
        if elite_k > 0:
            if (iteration + 1) % 3 == 0:
                refined = apply_abc(ants_paths[:elite_k], dist_matrix)
                ants_paths[:elite_k] = refined
                if refined[0][1] < best_path_length:
                    best_path, best_path_length = refined[0][0][:], refined[0][1]

        pheromone_matrix *= (1 - rho)
        for path, length in ants_paths[:max(len(ants_paths) // 2, elite_k)]:
            deposit = Q / (length + 1e-9)
            for i in range(len(path) - 1):
                pheromone_matrix[path[i], path[i + 1]] += deposit
                pheromone_matrix[path[i + 1], path[i]] += deposit

        print(f"Iteration {iteration + 1}: Best Path Length = {best_path_length:.4f}")
    return best_path, best_path_length


# -------------------------
# Configuration Wrappers
# -------------------------
def run_baseline_aco(intersections, dist_matrix):
    params = {'n_ants': 30, 'n_iterations': 50, 'Q': 100, 'rho': 0.3, 'alpha': 1.0, 'beta': 2.0, 'elite_k': 0}
    start = time.time()
    path, length = ant_colony_optimization_engine(intersections, dist_matrix, **params)
    duration = time.time() - start
    return path, length, duration


def run_hybrid_aco(intersections, dist_matrix):
    params = {'n_ants': 30, 'n_iterations': 50, 'Q': 100, 'rho': 0.3, 'alpha': 1.0, 'beta': 2.0, 'elite_k': 5}
    start = time.time()
    path, length = ant_colony_optimization_engine(intersections, dist_matrix, **params)
    duration = time.time() - start
    return path, length, duration


# -------------------------
# 🎨 Visualization Functions (FINAL UPDATES)
# -------------------------
# Node color set to '#555555' (Dark Grey) and size to 60.

NODE_COLOR_FINAL = '#006400'
NODE_SIZE_FINAL = 60


def visualize_single_route(intersections, labels, best_path=None, path_length=None, title="Route Visualization"):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#F9F9F9')
    ax.tick_params(axis='both', which='both', length=0)

    # Draw the Best Path (Route)
    if best_path is not None:
        path_x = intersections[best_path, 0]
        path_y = intersections[best_path, 1]

        ax.plot(path_x, path_y,
                color='#006B9A',
                linewidth=2.5,
                alpha=1.0,
                linestyle='-',
                marker='o', markersize=6, markerfacecolor='none', markeredgecolor='#006B9A', markeredgewidth=1.5,
                zorder=1, label='Optimal Route')

        # Highlight Start/End Point (Depot)
        start_node_index = best_path[0]
        ax.scatter(intersections[start_node_index, 0], intersections[start_node_index, 1],
                   color='#FF3B30', s=250, zorder=4, marker='*', edgecolor='white', linewidth=2,
                   label='Depot (Start/End)')

    # Draw all Intersections (Nodes) - Layered on top
    ax.scatter(intersections[:, 0], intersections[:, 1],
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

        # Draw Intersections (Nodes) on top
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
# Main Execution (RE-INCLUDED for completeness)
# -------------------------
if __name__ == "__main__":
    N_NODES = 100
    SEED = 48

    print(f"🚀 Running Comparison: {N_NODES} Intersections (Seed: {SEED})\n")

    random.seed(SEED)
    np.random.seed(SEED)
    nodes, labels = create_intersections(N_NODES)
    dists = compute_distance_matrix(nodes, peak_hour=True)

    print("1. Running Baseline ACO...")
    path_base, len_base, time_base = run_baseline_aco(nodes, dists)

    print("\n2. Running Hybrid ACO+ABC...")
    random.seed(SEED)
    np.random.seed(SEED)
    nodes_hyb, labels_hyb = create_intersections(N_NODES)
    dists_hyb = compute_distance_matrix(nodes_hyb, peak_hour=True)
    path_hyb, len_hyb, time_hyb = run_hybrid_aco(nodes_hyb, dists_hyb)

    print("\n" + "=" * 50)
    print("📊 FINAL COMPARISON")
    print("-" * 50)
    print(f"{'Algorithm':<20} | {'Best Length':<15} | {'Time (s)':<10}")
    print("-" * 50)
    print(f"{'ACO Only':<20} | {len_base:<15.4f} | {time_base:<10.2f}")
    print(f"{'Hybrid ACO+ABC':<20} | {len_hyb:<15.4f} | {time_hyb:<10.2f}")
    print("=" * 50 + "\n")

    # --- Generate the THREE images as requested ---

    print("--- Generating Individual Plots (Node Size 60, Dark Grey, Pop-Up Display) ---")

    # Image 1: Baseline ACO (Individual Plot)
    visualize_single_route(nodes, labels, path_base, len_base, "Baseline ACO Result")

    # Image 2: Hybrid ACO+ABC (Individual Plot)
    visualize_single_route(nodes_hyb, labels_hyb, path_hyb, len_hyb, "Hybrid ACO+ABC Result")

    print("\n--- Generating Side-by-Side Comparison Plot (Node Size 60, Dark Grey, Pop-Up Display) ---")

    # Image 3: Side-by-Side Comparison
    visualize_comparison_side_by_side(nodes, labels, path_base, len_base, "Baseline ACO",
                                      path_hyb, len_hyb, "Hybrid ACO+ABC")

    print("\nProcess complete.")