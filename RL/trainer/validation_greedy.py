"""Compatibility wrapper for the fixed training-rollout validation path."""


def validate_greedy(trainer_self):
    from trainer.validation_correct import validate_training_rollout

    return validate_training_rollout(trainer_self)
