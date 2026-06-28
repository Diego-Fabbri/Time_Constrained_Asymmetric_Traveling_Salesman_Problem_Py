# Asymmetric Traveling Salesman Problem with Time Windows (ATSPTW)

A **Mixed Integer Linear Programming (MILP)** model in **Python** for the **Asymmetric Traveling Salesman Problem with Time Windows**, built with the **[Pyomo](http://www.pyomo.org/)** optimization framework and solved via the **IBM ILOG CPLEX** solver.

## Overview

The Asymmetric Traveling Salesman Problem with Time Windows (ATSPTW) is a classic combinatorial optimization problem in Operations Research. A traveler starts from a depot, visits a set of nodes exactly once, and returns to the depot — respecting a time window at each node and an asymmetric travel time matrix (i.e., the travel time from node $i$ to node $j$ may differ from the travel time from $j$ to $i$).

The objective is **makespan minimization**: finding the route that minimizes the total time to complete the tour.

## Repository Contents

| File | Description |
|---|---|
| `Time_Constrained_Asymmetric_Traveling_Salesman_Problem.py` | Python script implementing and solving the ATSPTW via Pyomo and CPLEX |
| `Time_Constrained_Asymmetric_Traveling_Salesman_Problem_Results.txt` | Solver output: status, optimal makespan, visit times, and active arcs |
| `Asymmetric Traveling Salesman Problem with Time Windows_Math Formulation.pdf` | Mathematical formulation of the problem |

## Mathematical Formulation

### Sets

Let $G = (V, A)$ be a complete directed graph where:

- $V = \{0, 1, \dots, n\}$ = set of nodes (node $0$ is the depot)
- $A = \{(i,j) : i \neq j;\ i, j \in V\}$ = set of directed arcs

Every node $i \in V$ has an associated time window $TW_i = [l_i,\ u_i]$, where $l_i$ is the release time and $u_i$ is the deadline.

### Parameters

- $u_i$ = upper bound of the time window for node $i$; $\forall i \in V$
- $l_i$ = lower bound (release time) of the time window for node $i$; $\forall i \in V$
- $d_{ij}$ = distance from node $i$ to node $j$; $\forall (i,j) \in A$
- $v$ = travel speed
- $t_{ij} = d_{ij} / v$ = travel time from node $i$ to node $j$; $\forall (i,j) \in A$
- $M_{ij} = u_i - l_j + t_{ij}$ = big-M coefficient for the time-ordering constraints

### Variables

- $t_i$ = time instant at which node $i$ is visited; $\forall i \in V$
- $t_{n+1}$ = time at which the tour is completed (makespan)
- $y_{ij}$ = binary routing variable:

$$
y_{ij} = \begin{cases} 1 & \text{if the traveler goes directly from node } i \text{ to node } j \\ 0 & \text{otherwise} \end{cases}
$$

### Objective Function

**(1)** — Minimize the total tour completion time (makespan)

$$
\min \ t_{n+1}
$$

### Constraints

**(2)** — Visit time at each node is at least the travel time from the depot

$$
t_i \ge t_{0i} \cdot y_{0i} \qquad i = 1, 2, \dots, n
$$

**(3)** — Time ordering: if the traveler goes from $i$ to $j$, the visit time at $j$ must follow that at $i$ (big-M formulation)

$$
t_i - t_j + (u_i - l_j + t_{ij}) \cdot y_{ij} \le u_i - l_j \qquad \forall\, i, j = 1, \dots, n,\ i \neq j
$$

**(4)** — Each node is entered exactly once

$$
\displaystyle \sum_{i=0,\, i \neq j}^{n} y_{ij} = 1 \qquad j = 1, 2, \dots, n
$$

**(5)** — Each node is left exactly once

$$
\displaystyle \sum_{j=0,\, j \neq i}^{n} y_{ij} = 1 \qquad i = 1, 2, \dots, n
$$

**(6)** — The makespan is at least the return time from each node to the depot

$$
t_i + t_{i0} \le t_{n+1} \qquad i = 1, 2, \dots, n
$$

**(7)** — Time window feasibility at each node

$$
l_i \le t_i \le u_i \qquad i = 1, 2, \dots, n
$$

**(8)** — Exactly one arc enters the depot

$$
\displaystyle \sum_{i=1}^{n} y_{i0} = 1
$$

**(9)** — Exactly one arc leaves the depot

$$
\displaystyle \sum_{j=1}^{n} y_{0j} = 1
$$

**(10)** — Binary routing variables

$$
y_{ij} \in \{0,1\} \qquad \forall\, (i,j) \in A
$$

**(11)** — Non-negative visit times

$$
t_i \ge 0 \qquad \forall\, i = 0, 1, \dots, n+1
$$

A copy of this formulation is also available as a standalone PDF in this repository.

## Example Instance

The script ships with a sample instance featuring:

- **8 nodes** (1 depot + 7 nodes to visit)
- An **asymmetric** $8 \times 8$ distance matrix (self-loops set to 100,000 to exclude them from the solution)
- Travel speed: $v = 1$ (travel time equals distance)
- Time windows: lower bounds $l_i = 0$, upper bounds $u_i = 1000$ for all nodes
- Starting time at the depot: $t_0 = 0$

## Requirements

Install the required Python packages via pip:

```bash
pip install pyomo numpy pandas networkx
```

**IBM ILOG CPLEX** must also be installed separately on your system. An academic license is available free of charge through the [IBM Academic Initiative](https://www.ibm.com/academic).

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Diego-Fabbri/Time_Constrained_Asymmetric_Traveling_Salesman_Problem_Py.git
   cd Time_Constrained_Asymmetric_Traveling_Salesman_Problem_Py
   ```

2. Run the script:
   ```bash
   python Time_Constrained_Asymmetric_Traveling_Salesman_Problem.py
   ```

## Output

When executed, the script:
- Builds the MILP model using Pyomo's `ConcreteModel`
- Solves it via CPLEX and prints the full model structure to the console
- Writes the results to `Time_Constrained_Asymmetric_Traveling_Salesman_Problem_Results.txt`, including:
  - Solver status and termination condition
  - Optimal makespan $t_{n+1}$ (objective value)
  - Visit time $t_i$ at each node in tour order
  - Active arcs $y_{ij} = 1$ defining the optimal route
