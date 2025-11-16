# src/config.py

import os

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_OUTPUT = "outputs"
FIG_DIR = os.path.join(BASE_OUTPUT, "figures")
MODEL_DIR = os.path.join(BASE_OUTPUT, "models")
RESULT_DIR = os.path.join(BASE_OUTPUT, "results")

for d in [FIG_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)
