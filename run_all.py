"""Relance toutes les expériences, régénère les nombres du rapport et lance les tests."""
import subprocess
import sys

STEPS = [
    "experiments/markov_chain_example.py",
    "experiments/value_iteration_experiment.py",
    "experiments/main_qlearning.py",
    "experiments/gamma_experiment.py",
    "experiments/alpha_experiment.py",
    "experiments/epsilon_experiment.py",
    "report/make_results.py",
]

if __name__ == "__main__":
    for script in STEPS:
        print(f"==> {script}", flush=True)
        subprocess.run([sys.executable, script], check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)
