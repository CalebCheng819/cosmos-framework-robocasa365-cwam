import os

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_all_nano import (
    _action_policy_libero_nano_model_config,
    _action_policy_libero_s1_nano_model_config,
    action_policy_libero_all_nano,
    action_policy_libero_all_s1_nano,
)


def test_libero_s1_enables_fresh_router_without_mutating_baseline() -> None:
    baseline = action_policy_libero_all_nano
    s1 = action_policy_libero_all_s1_nano

    expected_tokenizer_path = os.environ["QWEN3_VL_TOKENIZER_PATH"]
    baseline_config = _action_policy_libero_nano_model_config()
    s1_config = _action_policy_libero_s1_nano_model_config()
    assert baseline_config["vlm_config"]["tokenizer"]["pretrained_model_name"] == expected_tokenizer_path
    assert s1_config["vlm_config"]["tokenizer"]["pretrained_model_name"] == expected_tokenizer_path
    assert baseline.model.config.get("action_relevance_enabled", False) is False
    assert s1.model.config.action_relevance_enabled is True
    assert s1.model.config.action_relevance_routing_dim == 512
    assert s1.model.config.action_relevance_num_heads == 8
    assert s1.model.config.action_relevance_topk == 32
    assert "action_relevance" not in baseline.optimizer.keys_to_select
    assert "action_relevance" in s1.optimizer.keys_to_select
    assert s1.optimizer.lr_multipliers.action_relevance == 5.0
    assert "action_relevance" in s1.checkpoint.keys_to_skip_loading
    assert "action2llm" in s1.checkpoint.keys_to_skip_loading
    assert "llm2action" in s1.checkpoint.keys_to_skip_loading
