from typing import Callable, Iterable, List

from continualworld.envs import get_cl_env, get_single_env
from continualworld.methods.vcl import VclMlpActor
from continualworld.sac.models import MlpActor
from continualworld.sac.utils.logx import EpochLogger
from continualworld.tasks import TASK_SEQS
from continualworld.utils.enums import BufferType
from continualworld.utils.run_utils import get_sac_class
from continualworld.utils.utils import get_activation_from_str
from input_args import cl_parse_args


def main(
    logger: EpochLogger,
    tasks: str,
    task_list: List[str],
    seed: int,
    steps_per_task: int,
    log_every: int,
    replay_size: int,
    batch_size: int,
    hidden_sizes: Iterable[int],
    buffer_type: str,
    reset_buffer_on_task_change: bool,
    reset_optimizer_on_task_change: bool,
    activation: Callable,
    use_layer_norm: bool,
    lr: float,
    gamma: float,
    alpha: str,
    target_output_std: float,
    cl_method: str,
    packnet_retrain_steps: int,
    regularize_critic: bool,
    cl_reg_coef: float,
    vcl_first_task_kl: bool,
    episodic_mem_per_task: int,
    episodic_batch_size: int,
    reset_actor_on_task_change: bool,
    reset_critic_on_task_change: bool,
    multihead_archs: bool,
    hide_task_id: bool,
    clipnorm: float,
    episodic_memory_from_buffer: bool,
    start_steps: int,
    exploration_kind: str,
    upload_weights: bool,
    freeze_actor_on_task_change: str,
    freeze_critic_on_task_change: str,
    transfer_alpha_on_task_change: bool,
):
    assert (tasks is None) != (task_list is None)
    if tasks is not None:
        tasks = TASK_SEQS[tasks]
    else:
        tasks = task_list
    train_env = get_cl_env(tasks, steps_per_task)
    # Consider normalizing test envs in the future.
    num_tasks = len(tasks)
    test_envs = [
        get_single_env(task, one_hot_idx=i, one_hot_len=num_tasks) for i, task in enumerate(tasks)
    ]
    steps = steps_per_task * len(tasks)

    if cl_method is not None:
        no_freezing = (freeze_actor_on_task_change is None and freeze_critic_on_task_change is None)
        assert no_freezing, "CL methods with freezing are not supported yet"

    num_heads = num_tasks if multihead_archs else 1
    actor_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        use_layer_norm=use_layer_norm,
        num_heads=num_heads,
        hide_task_id=hide_task_id,
    )
    critic_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        use_layer_norm=use_layer_norm,
        num_heads=num_heads,
        hide_task_id=hide_task_id,
    )

    if cl_method == "vcl":
        actor_cl = VclMlpActor
    else:
        actor_cl = MlpActor

    vanilla_sac_kwargs = {
        "env": train_env,
        "test_envs": test_envs,
        "logger": logger,
        "seed": seed,
        "steps": steps,
        "log_every": log_every,
        "replay_size": replay_size,
        "batch_size": batch_size,
        "actor_cl": actor_cl,
        "actor_kwargs": actor_kwargs,
        "critic_kwargs": critic_kwargs,
        "buffer_type": BufferType(buffer_type),
        "reset_buffer_on_task_change": reset_buffer_on_task_change,
        "reset_optimizer_on_task_change": reset_optimizer_on_task_change,
        "lr": lr,
        "alpha": alpha,
        "reset_actor_on_task_change": reset_actor_on_task_change,
        "reset_critic_on_task_change": reset_critic_on_task_change,
        "clipnorm": clipnorm,
        "gamma": gamma,
        "target_output_std": target_output_std,
        "start_steps": start_steps,
        "exploration_kind": exploration_kind,
        "upload_weights": upload_weights,
        "freeze_actor_on_task_change": freeze_actor_on_task_change,
        "freeze_critic_on_task_change": freeze_critic_on_task_change,
        "transfer_alpha_on_task_change": transfer_alpha_on_task_change,
    }

    sac_class = get_sac_class(cl_method)

    if cl_method is None:
        sac = sac_class(**vanilla_sac_kwargs)
    elif cl_method in ["l2", "ewc", "mas"]:
        sac = sac_class(
            **vanilla_sac_kwargs, cl_reg_coef=cl_reg_coef, regularize_critic=regularize_critic
        )
    elif cl_method == "vcl":
        sac = sac_class(
            **vanilla_sac_kwargs,
            cl_reg_coef=cl_reg_coef,
            regularize_critic=regularize_critic,
            first_task_kl=vcl_first_task_kl,
        )
    elif cl_method == "packnet":
        sac = sac_class(
            **vanilla_sac_kwargs,
            regularize_critic=regularize_critic,
            retrain_steps=packnet_retrain_steps,
        )
    elif cl_method == "agem":
        sac = sac_class(
            **vanilla_sac_kwargs,
            episodic_mem_per_task=episodic_mem_per_task,
            episodic_batch_size=episodic_batch_size,
        )
    elif cl_method == "episodic_replay":
        sac = sac_class(
            **vanilla_sac_kwargs,
            episodic_mem_per_task=episodic_mem_per_task,
            episodic_batch_size=episodic_batch_size,
            episodic_memory_from_buffer=episodic_memory_from_buffer,
            regularize_critic=regularize_critic,
            cl_reg_coef=cl_reg_coef,
        )
    else:
        raise NotImplementedError("This method is not implemented")
    sac.run()


if __name__ == "__main__":
    args = vars(cl_parse_args())
    logger = EpochLogger(args["logger_output"], config=args, group_id=args["group_id"])
    del args["group_id"]
    del args["logger_output"]
    main(logger, **args)
