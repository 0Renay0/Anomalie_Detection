from __future__ import annotations
import json 
import sys
import os
from typing import Any, Dict, List, Tuple
import numpy as np


#==================== CONFIG ====================
MODEL_PATH = "automate_thermostat.txt"
OUT_DIR = "data" 
OUT_DATASET = os.path.join(OUT_DIR, "dataset_mvp.npz")
OUT_Q2I = os.path.join(OUT_DIR, "q2i.json")


# Simulation parameters
DT = 0.01 # time step
T_Final = 60.0 # final time
NOISE_STD = 0.0 # noise

# Invariants 
DT_MIN_FACTOR = 1e-3 # min factor for dt/ minimal step time = DT * DT_MIN_FACTOR
MAX_HALVES = 20  # max number of halvings of the time step

# Window parameters
WIN = 8 # window length
HORIZON = 1 # prediction horizon
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
