import os
import optuna
from optuna.trial import TrialState

from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice,
    plot_rank,
    plot_param_importances,
)

STUDY_NAME = "npmpc_hopper_20260623_103500"
STORAGE = "postgresql:///optuna_db?host=%2Fvar%2Frun%2Fpostgresql"

OUTDIR = "optuna_figures/" + STUDY_NAME
os.makedirs(OUTDIR, exist_ok=True)

study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=STORAGE,
)

completed_trials = [
    trial for trial in study.trials if trial.state == TrialState.COMPLETE
]
print(f"Completed trials: {len(completed_trials)} / {len(study.trials)}")

plots = {
    "optimization_history": plot_optimization_history(study),
    "parallel_coordinate": plot_parallel_coordinate(study),
    "contour_lambda_sigma": plot_contour(study, params=["lambd", "sigma"]),
    "contour_lambda_td_slack": plot_contour(study, params=["lambd", "td_slack"]),
    "slice": plot_slice(study),
    "rank": plot_rank(study),
}

if len(completed_trials) >= 2:
    plots["param_importances"] = plot_param_importances(study)
else:
    print(
        "Skipping param_importances: Optuna needs at least two COMPLETE trials "
        f"but found {len(completed_trials)}.",
    )

for name, fig in plots.items():
    fig.write_html(os.path.join(OUTDIR, f"{name}.html"))

print(f"Saved HTML figures to {OUTDIR}/")
