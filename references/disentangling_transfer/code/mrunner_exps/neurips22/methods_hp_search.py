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
        "cl_method": [None],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["l2"],
        "cl_reg_coef": [1e3, 1e4, 1e5, 1e6],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["ewc"],
        "cl_reg_coef": [1e3, 1e4, 1e5, 1e6],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["mas"],
        "cl_reg_coef": [1e3, 1e4, 1e5, 1e6],
    },
    {
        "seed": list(range(10)),
        "cl_method": ["vcl"],
        "cl_reg_coef": [1e-2, 1e-1, 1.0, 10.0],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["packnet"],
        "packnet_retrain_steps": [50_000, 100_000, 200_000],
        "clipnorm": [2e-5],
    },
    {  # perfect memory
        "seed": list(range(20)),
        "cl_method": [None],
        "batch_size": [128, 256, 512],
        "buffer_type": ["reservoir"],
        "reset_buffer_on_task_change": [False],
        "replay_size": [20_000_000],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["agem"],
        "regularize_critic": [True],
        "episodic_mem_per_task": [10000],
        "episodic_batch_size": [128, 256],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["episodic_replay"],
        "cl_reg_coef": [10.0, 100.0, 1000.0],
        "episodic_batch_size": [128, 256],
        "episodic_mem_per_task": [10_000, 100_000],
        "clipnorm": [None, 0.01],
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
