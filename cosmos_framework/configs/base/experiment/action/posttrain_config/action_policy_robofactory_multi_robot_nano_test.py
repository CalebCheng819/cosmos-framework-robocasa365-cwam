from pathlib import Path


def test_offline_tokenizer_assets_are_environment_pinned() -> None:
    source = Path(__file__).with_name("action_policy_robofactory_multi_robot_nano.py").read_text()

    assert 'cfg["tokenizer"]["vae_path"] = "${oc.env:WAN_VAE_PATH}"' in source
    assert 'cfg["vlm_config"]["tokenizer"]["pretrained_model_name"] = "${oc.env:QWEN3_VL_TOKENIZER_PATH}"' in source
