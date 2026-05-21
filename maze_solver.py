import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


GRID = 5
START = (0, 0)
EXIT  = (4, 4)

# 1 = wall, 0 = free
MAZE = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0],
]

ACTIONS = {
    0: (-1,  0),  # UP
    1: ( 1,  0),  # DOWN
    2: ( 0, -1),  # LEFT
    3: ( 0,  1),  # RIGHT
}
ACTION_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
N_ACTIONS = 4
N_STATES  = GRID * GRID   # 25


def to_idx(r, c):
    return r * GRID + c


def step(state, action):
    r, c    = state
    dr, dc  = ACTIONS[action]
    nr, nc  = r + dr, c + dc

    # Out of bounds
    if not (0 <= nr < GRID and 0 <= nc < GRID):
        return state, -1, False

    # Wall
    if MAZE[nr][nc] == 1:
        return state, -10, False

    # Exit
    if (nr, nc) == EXIT:
        return (nr, nc), +100, True

    return (nr, nc), -1, False


def train(episodes=2000, alpha=0.1, gamma=0.99,
          epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
          max_steps=100):

    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards_log = []

    for ep in range(episodes):
        state        = START
        total_reward = 0

        for t in range(max_steps):
            s = to_idx(*state)

            # ε-greedy
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                action = int(np.argmax(Q[s]))

            next_state, reward, done = step(state, action)
            ns = to_idx(*next_state)

            # Bellman update
            Q[s, action] += alpha * (
                reward + gamma * np.max(Q[ns]) - Q[s, action]
            )

            state        = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_log.append(total_reward)

        if (ep + 1) % 200 == 0:
            avg = np.mean(rewards_log[-200:])
            print(f"  Episode {ep+1:>5} | ε={epsilon:.3f} | Avg Reward: {avg:.1f}")

    return Q


def find_path(Q, max_steps=50):
    state = START
    path  = [state]
    visited = set()

    for _ in range(max_steps):
        if state == EXIT:
            break
        if state in visited:
            print("  ⚠ Loop detected — maze may not be fully solved yet.")
            break
        visited.add(state)
        action     = int(np.argmax(Q[to_idx(*state)]))
        next_state, _, done = step(state, action)
        path.append(next_state)
        state = next_state
        if done:
            break

    return path


def draw(Q, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle("Q-Learning · Maze Solver", fontsize=15, fontweight="bold")

    ax.set_xlim(0, GRID); ax.set_ylim(0, GRID)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title("Maze — Learned Policy & Path")
    ax.set_xticks([]); ax.set_yticks([])

    for r in range(GRID):
        for c in range(GRID):
            if MAZE[r][c] == 1:
                color = "#2C3E50"
            elif (r, c) == EXIT:
                color = "#2ECC71"
            elif (r, c) == START:
                color = "#3498DB"
            else:
                color = "#ECF0F1"

            rect = patches.FancyBboxPatch(
                (c + 0.05, r + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.04",
                linewidth=0.5, edgecolor="#BDC3C7",
                facecolor=color)
            ax.add_patch(rect)

            if (r, c) == START:
                ax.text(c+0.5, r+0.5, "S", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="white")
            elif (r, c) == EXIT:
                ax.text(c+0.5, r+0.5, "E", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="white")
            elif MAZE[r][c] == 0:
                a   = int(np.argmax(Q[to_idx(r, c)]))
                sym = ACTION_SYMBOLS[a]
                ax.text(c+0.5, r+0.5, sym, ha="center", va="center",
                        fontsize=11, color="#7F8C8D")

    # Draw path
    if len(path) > 1:
        px = [c + 0.5 for (r, c) in path]
        py = [r + 0.5 for (r, c) in path]
        ax.plot(px, py, "o-", color="#E74C3C", linewidth=2.5,
                markersize=7, zorder=5, label="Optimal path")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("maze_solver_results.png", dpi=150, bbox_inches="tight")
    print("\n📊 Plot saved → maze_solver_results.png")
    plt.show()


def print_maze(path):
    path_set = set(path)
    print("\n  ── Maze (with optimal path marked as *) ──\n")
    for r in range(GRID):
        row = "  "
        for c in range(GRID):
            if (r, c) == START:
                row += " S "
            elif (r, c) == EXIT:
                row += " E "
            elif MAZE[r][c] == 1:
                row += " # "
            elif (r, c) in path_set:
                row += " * "
            else:
                row += " . "
        print(row)
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("  Q-Learning · Maze Solver")
    print("=" * 50)
    print(f"  Grid : {GRID}×{GRID}  |  Start: {START}  |  Exit: {EXIT}")
    print("=" * 50 + "\n")

    Q = train(
        episodes      = 2000,
        alpha         = 0.1,
        gamma         = 0.99,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.995,
        max_steps     = 100,
    )

    path = find_path(Q)

    print_maze(path)

    print("  ── Optimal Path ──")
    for i, (r, c) in enumerate(path):
        label = " ← START" if (r, c) == START else (" ← EXIT 🎯" if (r, c) == EXIT else "")
        print(f"  Step {i:>2}: ({r},{c}){label}")

    success = path[-1] == EXIT
    print(f"\n  Result: {'✅ Exit reached!' if success else '❌ Exit not reached'}")
    print(f"  Path length: {len(path)-1} steps\n")

    draw(Q, path)