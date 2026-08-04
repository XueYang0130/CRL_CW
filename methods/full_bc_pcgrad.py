from methods.full_bc import make_full_bc_method


METHOD = make_full_bc_method(
    "full_bc_pcgrad",
    gradient_strategy="pcgrad_sac_priority",
)
