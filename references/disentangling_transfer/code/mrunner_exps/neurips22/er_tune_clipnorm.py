from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "cl",
    "tasks": "CW20",
    "logger_output": ["tsv", "neptune"],
}
config = combine_config_with_defaults(config)

params_grid = [
    {
        "seed": list(range(20)),
        "cl_method": ["episodic_replay"],
        "cl_reg_coef": [100.0],
        "episodic_batch_size": [128],
        "episodic_mem_per_task": [10_000],
        "clipnorm": [None, 0.01, 0.1, 1.0, 10.0],
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
