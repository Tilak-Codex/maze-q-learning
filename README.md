# maze-q-learning

# 🧩 Q-Learning: Maze Solver

A simple and beginner-friendly implementation of **Q-Learning** applied to a **5×5 Maze** — no external RL libraries needed, just `numpy` and `matplotlib`.

---

## 🗺️ The Maze

```
 S  .  .  #  .
 .  #  .  .  .
 .  #  .  #  .
 .  .  .  #  .
 #  .  .  .  E
```

| Symbol | Meaning |
|--------|---------|
| `S` | Start position (0,0) |
| `E` | Exit / Goal (4,4) |
| `#` | Wall (blocked) |
| `.` | Free cell |

---

## 🧠 How Q-Learning Works

The agent starts with **zero knowledge** and learns by trial and error.

### Update Rule (Bellman Equation)
```
Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s',a') - Q(s,a) ]
```

| Symbol | Meaning |
|--------|---------|
| α | Learning rate (0.1) |
| γ | Discount factor (0.99) |
| ε | Exploration rate (1.0 → 0.01) |
| r | Reward received |

### Rewards
| Event | Reward |
|-------|--------|
| Reach exit | +100 |
| Hit a wall | -10 |
| Each step | -1 |

### Actions (4)
`UP ↑`, `DOWN ↓`, `LEFT ←`, `RIGHT →`

---

## 📁 Files

```
maze_solver.py           ← Main implementation (no external RL lib needed)
maze_solver_results.png  ← Training plots (generated on run)
README.md                ← This file
```

---

## 🚀 Getting Started

### Install
```bash
pip install numpy matplotlib
```

### Run
```bash
python maze_solver.py
```

### Output
```
==================================================
  Q-Learning · Maze Solver
==================================================
  Episode  200 | ε=0.364 | Avg Reward: -35.2
  Episode  400 | ε=0.133 | Avg Reward: 12.8
  ...
  Episode 2000 | ε=0.010 | Avg Reward: 74.1

  ── Maze (with optimal path marked as *) ──

   S  *  *  #  .
   *  #  .  .  .
   *  #  .  #  .
   *  *  *  #  .
   #  .  *  *  E

  Result: ✅ Exit reached!
  Path length: 10 steps
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| episodes | 2000 | Training episodes |
| alpha | 0.1 | Learning rate |
| gamma | 0.99 | Discount factor |
| epsilon | 1.0 → 0.01 | Exploration decay |
| max_steps | 100 | Steps per episode cap |

---

## 📚 References

- Sutton & Barto — *Reinforcement Learning: An Introduction* (2018)
- Watkins (1989) — *Learning from Delayed Rewards*
