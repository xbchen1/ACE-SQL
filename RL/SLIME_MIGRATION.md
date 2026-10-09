# Slime adapter status

The slime adapter is the current training entrypoint. See [RL/README.md](README.md) for setup and [paper alignment](../docs/paper_alignment.md) for the October 9, 2026 method contract.

The earlier adapter used majority-vote pruning and a target read before the current pool update. The current adapter follows the revised manuscript: union pruning and a target read after incorporating current execution-matched SQL. Its versioned reward state rejects incompatible old checkpoints.

The fixed H20/CUDA 12.9 environment and upstream metadata patch are retained. CPU behavior has been tested; a distributed GPU training and recovery run remains necessary before claiming runtime reproduction of the manuscript.
