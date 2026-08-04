from methods.full_bc import make_full_bc_method


METHOD = make_full_bc_method(
    "full_bc_norm_balanced",
    gradient_strategy="norm_balanced",
)
