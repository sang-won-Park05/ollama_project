# Training Workflow

`train/`은 EXAONE 3.5 7.8B 계열 모델에 대해 보안 거부/요약 행동을 학습시키기 위한 QLoRA 스타일 학습 코드를 담는다.

권장 순서:

1. `python train/inspect_modules.py --config configs/train_config_lora.yaml`
2. `python data/scripts/make_dataset.py --config configs/project_config.yaml`
3. `python train/train_lora.py --config configs/train_config_lora.yaml`
4. 기본 파이프라인에서는 `test_adapter.py`를 건너뛰고 `eval/run_attack_eval.py`, `eval/run_benign_eval.py`, `eval/compare_results.py`로 진행
5. 선택적 디버깅이 필요할 때만 `python train/test_adapter.py --config configs/eval_config.yaml --profile lora`
6. 필요 시 `python train/merge_adapter.py --config configs/train_config_lora.yaml`
7. 추후 export 준비는 `python train/export_artifacts.py --config configs/train_config_lora.yaml`

주의:

- base model은 `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`를 기본값으로 가정한다.
- `trust_remote_code=True`가 필요하다.
- RTX 5090급 단일 GPU를 가정한 4bit + bf16 + gradient accumulation 설정이다.
- 실제 학습 전에 `outputs/`, `logs/`, `data/processed/` 경로가 준비되어 있는지 확인한다.
