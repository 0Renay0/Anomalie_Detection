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

