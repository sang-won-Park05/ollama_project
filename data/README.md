# Data Layout

`data/`는 전부 로컬 방어 실험용 자산만 포함한다. 실제 운영 문서나 실제 자격증명은 포함하지 않는다.

- `raw/`: fake secret가 섞인 실험 문서
- `processed/`: 학습/평가용 JSONL 및 통계
- `templates/`: 시스템 프롬프트, 거부 템플릿, 정상 응답 템플릿
- `scripts/`: 데이터셋 생성/검증 코드

권장 흐름:

1. `raw/` 내용을 검토해 모두 fake 값인지 확인한다.
2. `python data/scripts/make_dataset.py --config configs/project_config.yaml` 실행
3. `python data/scripts/validate_dataset.py --config configs/project_config.yaml` 실행
4. `processed/` 산출물로 학습/평가를 연결한다.
