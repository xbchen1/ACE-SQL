.PHONY: check-paths syntax

check-paths:
	@bash scripts/check_paths.sh

syntax:
	@bash -n scripts/train_rl_pcgrad.sh
	@python3 -m py_compile trainer/main_two_pass.py trainer/two_pass_ray_trainer.py rewards/joint_reward.py rewards/verl_reward_wrapper.py utils/path_utils.py
