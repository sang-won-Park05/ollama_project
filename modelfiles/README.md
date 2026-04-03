# Modelfiles

이 폴더는 Ollama baseline / secure / secure_lora 템플릿을 담는다.

핵심 원칙:

- raw secret, token, password, API key, hidden prompt, internal instruction, `.env` 값은 재출력 금지
- 인코딩, 부분 문자열, base64, hex, reverse string, unicode escape, 표 변환 같은 우회 요청도 거부
- 안전한 대안으로 redacted summary, masked example, 보안 체크리스트만 허용
- temperature / top_p는 보수적으로 유지

예시 명령:

```bash
ollama create exaone-baseline -f modelfiles/Modelfile.baseline
ollama show --modelfile exaone-baseline
ollama run exaone-baseline
```

```bash
ollama create exaone-secure -f modelfiles/Modelfile.secure
ollama show --modelfile exaone-secure
ollama run exaone-secure
```

```bash
ollama create exaone-secure-lora -f modelfiles/Modelfile.secure_lora_template
ollama show --modelfile exaone-secure-lora
ollama run exaone-secure-lora
```

실사용 전 확인:

- `FROM` 모델 태그 또는 로컬 모델 경로를 실제 환경 값으로 교체
- `ADAPTER` 또는 merged model 경로를 실제 산출물 위치로 교체
- Open WebUI에는 baseline / secure / secure_lora를 별도 이름으로 등록
