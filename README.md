# ollama_project

`ollama_project`는 EXAONE 3.5 7.8B 계열 모델을 대상으로 Open WebUI + Ollama 환경에서 프롬프트 인젝션, 역할 혼동, 문서 기반 간접 유출, 부분 문자열/인코딩 우회 같은 공격에 대한 방어 실험을 재현하기 위한 로컬 프로젝트 초안이다.

이 저장소는 다음을 포함한다.

- 공격 프롬프트 세트
- dummy secret / dummy env / dummy internal docs
- 안전 응답 데이터셋 생성 스크립트
- QLoRA 스타일 보안 LoRA 학습 코드
- 어댑터 테스트 코드
- 공격/정상 평가 코드
- secret-like pattern 감지 로직
- Ollama baseline / secure / secure_lora Modelfile
- 설정 파일과 실행 순서 문서

## 경고

- 실제 API 키, 실제 비밀번호, 실제 `.env` 비밀값, 실제 자격증명을 절대 넣지 말 것.
- 이 프로젝트의 모든 비밀값 예시는 fake / dummy 값만 사용한다.
- 공격 프롬프트는 오직 로컬 보안 평가 및 방어 실험 목적이다.
- Open WebUI 연동 전후로도 실제 프로덕션 문서나 실제 내부 자격증명을 마운트하지 말 것.

## 실험 범위

- base model: `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`
- runtime: Hugging Face Transformers + PEFT + TRL + Ollama
- 목표: prompt injection, role confusion, indirect document leakage, unicode/encoding based bypass 시도에 대한 방어 성능 측정
- 출력: dataset, adapter, attack eval report, benign eval report, comparison summary, secure Modelfile

## 폴더 구조 설명

- `attacks/`: 카테고리별 공격 프롬프트와 메타데이터
- `data/raw/`: 전부 dummy 값으로만 구성된 테스트 문서
- `data/processed/`: 학습/평가용 JSONL 산출물
- `data/templates/`: 시스템 프롬프트와 안전 응답 템플릿
- `data/scripts/`: 데이터셋 생성 및 검증 스크립트
- `train/`: LoRA 학습, 모듈 점검, 병합, 어댑터 테스트 코드
- `eval/`: HF/Ollama 기반 공격 평가와 결과 비교 코드
- `modelfiles/`: Ollama baseline / secure / secure_lora 템플릿
- `logs/`: 학습/평가/디버그 로그 산출 위치
- `outputs/`: adapter, merged model, gguf, checkpoint 산출 위치
- `notebooks/`: 결과 탐색용 노트북 초안

## 설치 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

GPU 환경이 없으면 학습과 HF 대규모 평가 실행은 생략하고 데이터셋 생성 및 정적 검증만 수행해도 된다.

## dummy 데이터 확인 / 검증

raw 문서는 이미 포함되어 있으며 모두 fake 값만 사용한다.

```bash
python data/scripts/make_dataset.py --config configs/project_config.yaml
python data/scripts/build_attack_eval_set.py --config configs/project_config.yaml
python data/scripts/build_benign_eval_set.py --config configs/project_config.yaml
python data/scripts/validate_dataset.py --config configs/project_config.yaml
```

## baseline 공격 평가

Ollama에 baseline 모델을 먼저 생성한다.

```bash
ollama create exaone-baseline -f modelfiles/Modelfile.baseline
python eval/run_attack_eval.py --config configs/eval_config.yaml --profile baseline
```

## secure Modelfile 평가

```bash
ollama create exaone-secure -f modelfiles/Modelfile.secure
python eval/run_attack_eval.py --config configs/eval_config.yaml --profile secure_modelfile
```

## LoRA 학습

먼저 모듈 이름을 확인해 target module 후보를 검토한다.

```bash
python train/inspect_modules.py --config configs/train_config.yaml
```

이후 LoRA 학습을 실행한다.

```bash
python train/train_lora.py --config configs/train_config.yaml
```

## adapter 테스트

```bash
python train/test_adapter.py --config configs/eval_config.yaml
```

## before / after 비교

```bash
python eval/compare_results.py --config configs/eval_config.yaml
```

## Open WebUI 연동 시 주의점

- Open WebUI의 `Documents`, `Knowledge`, `Pipelines`에 실제 운영 문서를 넣지 말 것.
- 테스트용 문서 마운트 경로와 실제 운영 데이터 경로를 분리할 것.
- secure Modelfile 또는 LoRA 적용 모델을 별도 태그로 등록하고 baseline 모델과 혼용하지 말 것.
- Open WebUI conversation export 기능을 켠 경우 로그에 민감한 실험 산출물이 쌓이지 않도록 저장 위치를 분리할 것.
- RAG 실험 시 검색 문서에 포함된 fake secret이 그대로 answer grounding 되지 않는지 별도로 검증할 것.

## 한계점

- 규칙 기반 pattern 탐지는 모든 유출 변형을 잡아내지 못한다.
- secure Modelfile만으로는 긴 컨텍스트 오염과 강한 role confusion을 완전히 막기 어렵다.
- QLoRA는 데이터 품질과 target module 선택에 크게 의존한다.
- Ollama / tokenizer / template 차이로 HF 로컬 결과와 Ollama 결과가 완전히 일치하지 않을 수 있다.

## 향후 확장

- 이미지 OCR이 포함된 실제 multimodal 모델 평가
- retrieval stage sanitization 및 document redaction 파이프라인 추가
- judge model 기반 semantic leakage scoring 추가
- Open WebUI pipeline hook을 통한 pre-filter / post-filter 통합
- GGUF export 자동화 및 모델 카드 생성

## 권장 실험 순서

1. dummy data 확인
2. baseline 공격 평가
3. secure Modelfile 적용 후 재평가
4. LoRA 학습
5. adapter 테스트
6. 재평가
7. 결과 비교
