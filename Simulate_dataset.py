from __future__ import annotations
import json 
import sys
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt  

# --- Plotting config ---
SHOW_PLOTS = True    # afficher à l'écran (plt.show())
SAVE_PLOTS = True    # sauvegarder des images dans OUT_DIR


#==================== CONFIG ====================
MODEL_PATH = "automate_thermostat.txt"
OUT_DIR = "data" 
OUT_DATASET = os.path.join(OUT_DIR, "dataset_mvp.npz")
OUT_Q2I = os.path.join(OUT_DIR, "q2i.json")


# Simulation parameters
DT = 0.001 # time step
T_FINAL = 60.0 # final time
NOISE_STD = 0.0 # noise

# Invariants 
DT_MIN_FACTOR = 1e-3 # min factor for dt/ minimal step time = DT * DT_MIN_FACTOR
MAX_HALVES = 20  # max number of halvings of the time step

# Window parameters
WIN = 8 # window length
HORIZON = 1 # prediction horizon

# Plotting config 
SHOW_PLOTS = True    # afficher à l'écran (plt.show())
SAVE_PLOTS = True    # sauvegarder des images dans OUT_DIR
# ===========================================================

class SimError(Exception):
    """ Error during simulation """
    
def bind_functions(func_src: Dict[str, str]) -> Dict[str, Any]:
    
    ns: Dict[str, Any] = {}
    for name, src in func_src.items():
        exec(compile(src, filename=name + ".py", mode="exec"), {"__builtins__": {}}, ns)
        if name not in ns or not callable(ns[name]):
            raise ValueError(f"Function {name} not defined in source")
    return ns

def step_euler(x: np.ndarray, f, t: float, dt: float) -> np.ndarray:
    """ Perform one Euler step """
    dx = np.array(f(x.tolist(), t), dtype=float)
    return x + dt * dx

def guard_true(ns: Dict[str, Any], guard_name: str, x: np.ndarray) -> bool:
    """ Evaluate guard function """
    return bool(ns[guard_name](x.tolist()))

def reset(ns: Dict[str, Any], reset_name: str | None, x: np.ndarray) -> np.ndarray:
    """ Apply reset function if provided, else return x unchanged """
    if not reset_name:
        return x
    out = ns[reset_name](x.tolist())
    return np.array(out, dtype=float)

def inv_ok(ns: Dict[str, Any], inv_name: str, x: np.ndarray) -> bool:
    """ Evaluate invariant function """
    return bool(ns[inv_name](x.tolist()))

def basic_model_checks(data: Dict[str, Any]) -> None:
    """ Basic checks on the model structure """
    required = ["Q", "X", "q0", "x0", "flow", "Inv"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SimError("Missing fields in model: " + ", ".join(missing))

    Q: List[str] = data["Q"]
    X: List[str] = data["X"]
    q0: str = data["q0"]
    x0: List[float] = data["x0"]
    flow = data["flow"]
    Inv = data["Inv"]

    if not (isinstance(Q, list) and len(Q) > 0 and all(isinstance(q, str) for q in Q)):
        raise SimError("Q must be a non-empty list of strings.")

    if not (isinstance(X, list) and len(X) > 0 and all(isinstance(x, str) for x in X)):
        raise SimError("X must be a non-empty list of strings.")

    if not isinstance(q0, str) or q0 not in Q:
        raise SimError("q0 must be a string belonging to Q.")

    if not (isinstance(x0, list) and len(x0) == len(X) and all(isinstance(v, (int, float)) for v in x0)):
        raise SimError("x0 must be a list of numbers (int or float) of the same length as X.")

    if not (isinstance(flow, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in flow.items())
            and all(k in flow for k in Q)):
        raise SimError("flow must be a dictionary {state: function_name} for each state in Q.")

    if not (isinstance(Inv, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in Inv.items())
            and all(k in Inv for k in Q)):
        raise SimError("Inv must be a dictionary {state: function_name} for each state in Q.")

    if "Guard" in data:
        Guard = data["Guard"]
        if not isinstance(Guard, dict):
            raise SimError("'Guard' must be an object {EtatSource: {EtatDest: guard_name}}.")
    if "Jump" in data and not isinstance(data["Jump"], dict):
        raise SimError("'Jump' must be an object {EtatSource: {EtatDest: reset_name}}.")
    if "T" in data and not isinstance(data["T"], list):
        raise SimError("'T' must be a list of transition objects.")
    
    
def simulate(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Simulate the hybrid automaton and generate dataset using Invs + guards + resets
    
    Returns:
        - times: (N,)
      - xs:    (N, dim_x)
      - qs:    (N,) indices des modes
      - q2i:   mapping {nom_mode: index}
    """
    basic_model_checks(data)
    
    Q: List[str] = data["Q"]
    X: List[str] = data["X"]
    q0: str = data["q0"]
    x0: List[float] = data["x0"]
    
    flow = data["flow"]
    Inv = data["Inv"]
    Guard = data.get("Guard", {})
    Jump = data.get("Jump", {})
    
    ns = bind_functions(data["functions"])
    
    # Encodings for modes : Q1 -> 0, Q2 -> 1 ..
    q2i: Dict[str, int] = {q: i for i, q in enumerate(Q)}
    
    x = np.array(x0, dtype=float)
    q = q0
    t = 0.0

    times = [t]
    xs = [x.copy()]
    qs = [q2i[q]]
    
    dt_max = DT
    dt_min = DT * DT_MIN_FACTOR
    
    while t < T_FINAL - 1e-12:
        dt = dt_max
        step_accepted = False

        for _ in range(MAX_HALVES + 1):
            # Try integration in the current mode on [t, t+dt]
            f = ns[flow[q]]
            x_trial = step_euler(x, f, t, dt)

            # Check the invariant of the current mode at the attempted point
            if inv_ok(ns, Inv[q], x_trial):
                q_new = q

                # Check the guards at the attempted point
                transitioned = False
                if q in Guard:
                    # Simple priority scheme: first guard found is taken
                    for qdst, gname in Guard[q].items():
                        if guard_true(ns, gname, x_trial):
                            # Guard activated, perform the jump 
                            rname = Jump.get(q, {}).get(qdst, None)
                            x_after = reset(ns, rname, x_trial)

                            # Check the invariant of the destination mode after reset
                            if not inv_ok(ns, Inv[qdst], x_after):
                                raise SimError(
                                    f"Invariant of destination mode '{qdst}' violated at t={t+dt:.6f}"
                                )

                            q_new = qdst
                            x_trial = x_after
                            transitioned = True
                            break

                # Step accepted
                t = t + dt
                x = x_trial
                q = q_new

                # Add noise Optionally
                x_obs = x + (np.random.normal(0.0, NOISE_STD, size=x.shape) if NOISE_STD > 0 else 0.0)

                times.append(t)
                xs.append(np.array(x_obs, dtype=float))
                qs.append(q2i[q])
                step_accepted = True
                break
            else:
                # Invariant violated without activatable guard → reduce step (bisection)
                dt *= 0.5
                if dt < dt_min:
                    raise SimError(
                        f"Invariant of mode '{q}' violated (unable to find valid sub-step) towards t≈{t:.6f}"
                    )

        if not step_accepted:
            raise SimError("No sub-step accepted (conflict invariants/guards).")

    return np.array(times), np.vstack(xs), np.array(qs, dtype=int), q2i


def plot_simulation(times: np.ndarray,
                    xs: np.ndarray,
                    qs: np.ndarray,
                    q2i: Dict[str, int],
                    var_names: List[str],
                    out_dir: str) -> None:
    """
    Plot the simulation results: continuous states and discrete modes over time.
    1 plot for continuous states, 1 plot for modes.
    """
    os.makedirs(out_dir, exist_ok=True)
    i2q = {i: q for q, i in q2i.items()}

    # -------- Figure 1: continuous states --------
    plt.figure(figsize=(8, 4))
    for k in range(xs.shape[1]):
        plt.plot(times, xs[:, k], label=var_names[k] if k < len(var_names) else f"x{k}")
    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.title("Continuous states over time")
    plt.legend(loc="best")
    plt.grid(True)
    if SAVE_PLOTS:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sim_states.png"), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # -------- Figure 2: discrete modes --------
    plt.figure(figsize=(8, 3))
    plt.step(times, qs, where="post")
    plt.xlabel("Time")
    plt.ylabel("Mode index")
    plt.title("Discrete modes over time")
    # Y ticks avec noms de modes
    yticks = sorted(set(qs.tolist()))
    yticklabels = [i2q[i] for i in yticks]
    plt.yticks(yticks, yticklabels)
    plt.grid(True)
    if SAVE_PLOTS:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sim_modes.png"), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def make_windows(times: np.ndarray,
                 xs: np.ndarray,
                 qs: np.ndarray,
                 win: int = WIN,
                 horizon: int = HORIZON):
    """
    Construit des paires (entrée -> cibles) pour l'entraînement:
      - entrée  : fenêtre (win) sur x et q
      - cibles  : x à +horizon et prochain mode (à +1)
    """
    Xwin, Qwin, yx, yq = [], [], [], []
    T = len(times)
    for i in range(0, T - win - horizon):
        Xwin.append(xs[i:i + win, :])
        Qwin.append(qs[i:i + win])
        yx.append(xs[i + win + horizon - 1, :])  # predict x at horizon
        yq.append(qs[i + win])                    # next mode
    return np.array(Xwin), np.array(Qwin), np.array(yx), np.array(yq)


def main() -> None:
    # Load model
    try:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}", file=sys.stderr)
        sys.exit(2)
        
    # Simulate
    try:
        times, xs, qs, q2i = simulate(data)
    except SimError as e:
        print(f"[ERREUR SIMU] {e}", file=sys.stderr)
        sys.exit(3)
    # Plot results
    try:
        plot_simulation(times, xs, qs, q2i, var_names=data.get("X", []), out_dir=OUT_DIR)
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", file=sys.stderr)
        
    # Create windows
    Xwin, Qwin, yx, yq = make_windows(times, xs, qs, win=WIN, horizon=HORIZON)
    if Xwin.size == 0:
        print("[ERREUR] Fenêtrage vide. Allonge T_FINAL, réduis WIN/HORIZON, ou augmente DT.", file=sys.stderr)
        sys.exit(4)
        
    
if __name__ == "__main__":
    main()
