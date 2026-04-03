# Evaluation Workflow

`eval/`은 공격 프롬프트와 정상 프롬프트를 Hugging Face 로컬 모델 또는 Ollama 백엔드로 실행하고, secret-like 패턴 누설 여부를 정량화하는 코드 모음이다.

핵심 흐름:

1. `python data/scripts/build_attack_eval_set.py --config configs/project_config.yaml`
2. `python data/scripts/build_benign_eval_set.py --config configs/project_config.yaml`
3. `python eval/run_attack_eval.py --config configs/eval_config.yaml --profile baseline`
4. `python eval/run_attack_eval.py --config configs/eval_config.yaml --profile secure_modelfile`
5. `python eval/run_attack_eval.py --config configs/eval_config.yaml --profile lora`
6. `python eval/run_benign_eval.py --config configs/eval_config.yaml --profile lora`
7. `python eval/compare_results.py --config configs/eval_config.yaml`

평가 리포트는 기본적으로 `logs/eval/` 아래 JSON으로 저장된다.
