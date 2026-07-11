from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "cl",
    "tasks": "CW10",
    "logger_output": ["tsv", "neptune"],
}
config = combine_config_with_defaults(config)

params_grid = [
    {  # fine-tuning
        "seed": list(range(20)),
        "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "cl_method": [None],
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="anonymized-2",
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    exclude=[".neptune"],
)
