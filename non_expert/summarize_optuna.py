# summarize_optuna.py
import optuna
from optuna.trial import TrialState

STUDY_NAME = "npmpc_swimmer"
STORAGE = "postgresql:///optuna_db?host=%2Fvar%2Frun%2Fpostgresql"

study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=STORAGE,
)

trials = study.trials
completed = [t for t in trials if t.state == TrialState.COMPLETE]
pruned = [t for t in trials if t.state == TrialState.PRUNED]
failed = [t for t in trials if t.state == TrialState.FAIL]

print(f"Study: {STUDY_NAME}")
print(f"Total trials: {len(trials)}")
print(f"Completed: {len(completed)}")
print(f"Pruned: {len(pruned)}")
print(f"Failed: {len(failed)}")

print("\nBest trial:")
print(f"  Number: {study.best_trial.number}")
print(f"  Value:  {study.best_value}")
print("  Params:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")

df = study.trials_dataframe()
df.to_csv("optuna_trials_summary.csv", index=False)
print("\nSaved: optuna_trials_summary.csv")

cols = ["number", "value", "state"] + [c for c in df.columns if c.startswith("params_")]
print("\nTop 10 completed trials:")
print(
    df[df["state"] == "COMPLETE"]
    .sort_values("value", ascending=False)
    [cols]
    .head(10)
    .to_string(index=False)
)