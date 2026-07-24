# ClonEx-SAC Alignment

The PyTorch implementation follows the final ClonEx-SAC method described in
the NeurIPS 2022 paper and its released TensorFlow source:

- shared four-layer actor and critic backbones with one output head per task;
- task one-hot used for head routing and hidden from backbone inputs;
- actor-only behavioral cloning from episodic memory;
- `10_000` episodic examples retained per completed task;
- episodic batch size `128` and actor cloning coefficient `100`;
- no continual-learning regularization for either critic;
- best-return exploration during the first `10_000` steps of each new task;
- global gradient norm clipping at `0.1`;
- online replay and optimizer reset at each task boundary.

The critic is still trained normally with the SAC Bellman objective. "No
critic regularization" means that no episodic distillation loss is added to
that objective.

The experiment protocol retains the original SAC defaults: four hidden layers
of width `256`, batch size `128`, replay size `1_000_000`, learning rate
`1e-3`, discount `0.99`, Polyak coefficient `0.995`, target output standard
deviation `0.089`, one million steps per task, and episode length `200`.

This project intentionally uses PyTorch and Meta-World v3 observations. It
therefore reproduces the method and training protocol, but it is not a
bit-for-bit reproduction of the paper's TensorFlow, old Meta-World, and
MuJoCo 2.0 runtime.
