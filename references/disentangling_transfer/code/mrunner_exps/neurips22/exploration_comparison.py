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
    {  # fine-tuning
        "seed": list(range(20)),
        "cl_method": [None],
        "exploration_kind": [
            None,
            "previous",
            "uniform_previous",
            "uniform_previous_or_current",
            "best_return",
            "softmax_return_1.0",
        ],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["l2"],
        "cl_reg_coef": [1e5],
        "exploration_kind": [
            None,
            "previous",
            "uniform_previous",
            "uniform_previous_or_current",
            "best_return",
            "softmax_return_1.0",
        ],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["ewc"],
        "cl_reg_coef": [1e4],
        "exploration_kind": [
            None,
            "previous",
            "uniform_previous",
            "uniform_previous_or_current",
            "best_return",
            "softmax_return_1.0",
        ],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["packnet"],
        "packnet_retrain_steps": [100_000],
        "clipnorm": [2e-5],
        "exploration_kind": [
            None,
            "previous",
            "uniform_previous",
            "uniform_previous_or_current",
            "best_return",
            "softmax_return_1.0",
        ],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["episodic_replay"],
        "cl_reg_coef": [100.0],
        "episodic_batch_size": [128],
        "episodic_mem_per_task": [10_000],
        "clipnorm": [None, 0.01, 0.1],
        "exploration_kind": [
            None,
            "previous",
            "uniform_previous",
            "uniform_previous_or_current",
            "best_return",
            "softmax_return_1.0",
        ],
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
