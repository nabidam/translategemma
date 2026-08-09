"""Configuration helpers for bounded, real training canary runs."""

import copy


def canary_run_config(config):
    """Return an isolated config and limits for a real, bounded canary run."""
    canary = config.get("canary")
    if not isinstance(canary, dict):
        raise ValueError("config must contain a canary section")

    max_examples = canary.get("max_examples")
    if not isinstance(max_examples, int) or isinstance(max_examples, bool) or max_examples <= 0:
        raise ValueError("canary.max_examples must be a positive integer")
    max_steps = canary.get("max_steps")
    if max_steps is not None and (
        not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
    ):
        raise ValueError("canary.max_steps must be null or a positive integer")

    output_dir = canary.get("output_dir")
    if not output_dir:
        raise ValueError("canary.output_dir must be configured")
    evaluation_output_dir = canary.get("evaluation_output_dir")
    if canary.get("run_after_training") and not evaluation_output_dir:
        raise ValueError(
            "canary.evaluation_output_dir must be configured when canary.run_after_training is true"
        )

    canary_config = copy.deepcopy(config)
    canary_config["model"]["output_dir"] = str(output_dir)
    canary_config["training"]["resume_from_checkpoint"] = canary.get("resume_from_checkpoint")
    canary_config["evaluation"]["run_after_training"] = bool(canary.get("run_after_training", False))
    if evaluation_output_dir:
        canary_config["evaluation"]["output_dir"] = str(evaluation_output_dir)
    return canary_config, max_examples, max_steps
