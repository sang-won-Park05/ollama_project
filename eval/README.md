# Evaluation Workflow

`eval/`은 공격 프롬프트와 정상 프롬프트를 Hugging Face 로컬 모델 또는 Ollama 백엔드로 실행하고, secret-like 패턴 누설 여부를 정량화하는 코드 모음이다.

핵심 흐름:

1. `python data/scripts/build_attack_eval_set.py --config configs/project_config.yaml`
2. `python data/scripts/build_benign_eval_set.py --config configs/project_config.yaml`
3. `python eval/run_attack_eval.py --config configs/eval_config.yaml --profile baseline`
4. `python eval/run_attack_eval.py --config configs/eval_config.yaml --profile secure_modelfile`
5. 빠른 확인은 `python eval/run_attack_eval.py --config configs/eval_config.yaml --profile lora_fast`
6. 전체 LoRA 공격 평가는 `python eval/run_attack_eval.py --config configs/eval_config.yaml --profile lora`
7. 전체 LoRA 정상 평가는 `python eval/run_benign_eval.py --config configs/eval_config.yaml --profile lora`
8. 결과 비교는 `python eval/compare_results.py --config configs/eval_config.yaml`

평가 리포트는 기본적으로 `logs/eval/` 아래 JSON으로 저장된다.

`run_attack_eval.py`는 각 attack 시작/종료 시점을 출력하고, attack별 소요 시간을 기록하며, attack마다 결과를 중간 저장한다. `resume: true` profile은 기존 결과 파일이 있으면 완료된 attack을 건너뛰고 이어서 실행한다.
