# BiRefNet 저장소 감사

2026년 10월 13일 기준, `0573736`에서 시작한 전체 저장소 분석과 수정 결과다.
API 자원 고갈 경로, 학습 손실과 설정 검증, 체크포인트 재개, 의존성 보안을
점검했다. 아래 코드 위치는 이번 수정 후 파일의 줄 번호다. GPU 처리량이나
실제 SEM 데이터의 모델 정확도를 평가한 결과는 아니다.

## 심각도별 수정 사항

P1은 서비스 중단·가중치 손상·코드 실행 위험, P2는 조건부 오류 또는 의존성
위험, P3는 낮은 영향의 설정 검증 문제다. 외부 공격 가능성은 API 노출 및
체크포인트 출처에 따라 달라진다.

| 우선순위 | 원인과 영향 | 수정과 근거 | 검증 |
| --- | --- | --- | --- |
| P1 | HTTP 본문 전체를 JSON으로 읽은 뒤에야 base64 길이를 검사했다. 큰 JSON이나 알 수 없는 필드만으로 메모리를 소비할 수 있었다. | [backend/app.py:16](backend/app.py#L16): ASGI 수신 단계에서 본문을 누적 검사한다. base64 상한 + 64 KiB 초과 시 413. 청크별 객체를 보관하지 않고 bytearray를 사용한다. | [test_api_predict.py:188](tests/test_api_predict.py#L188): 잘못된 JSON도 크기 초과를 먼저 거부. Content-Length 없는 청크, 허용 경계의 본문 재생도 검사. |
| P1 | `tiles`의 양의 정수에 상한이 없어 `grid²`개의 박스와 추론 작업을 요청할 수 있었다. | [backend/schema.py:24](backend/schema.py#L24): 엄격한 JSON 정수 및 `sum(grid²) <= 64` 검사. 숫자 문자열·실수·bool과 작업량 초과를 422로 거부한다. | [test_api_predict.py:86](tests/test_api_predict.py#L86), [허용 경계:221](tests/test_api_predict.py#L221): 65패치 거부, 64패치 허용. |
| P1 | 이미지 픽셀 제한만으로는 클래스 수에 비례하는 logits 메모리를 제한하지 못했다. 36M픽셀 × 256클래스 × 4바이트는 단일 배열만 36.864 GB다. | [backend/route.py:32](backend/route.py#L32): 원본 해상도 logits 배열의 256 MiB 예산을 추론 전에 검사하고 초과 시 413. | [test_api_predict.py:179](tests/test_api_predict.py#L179): 4·256클래스 초과 입력에서 추론이 실행되지 않는지 검사. 예산 경계는 허용. |
| P1 | 무제약 `torch` 의존성은 `weights_only=True` 역직렬화 취약점이 있는 구버전 설치도 허용했다. 현재 설치된 2.14는 해당 영향 범위 밖이었다. | [requirements.txt:1](requirements.txt#L1): `torch>=2.10.0`. [PyTorch 공식 공지](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)는 <=2.9.1의 악성 체크포인트 위험과 수정 버전을 명시한다. | 의존성 검사 및 기존 엄격한 체크포인트 호환성 테스트 통과. 악성 체크포인트 실행은 하지 않았다. |
| P1 | AdamW는 `weight_decay=+Inf`를 허용했다. 첫 업데이트의 감쇠 연산으로 학습 가중치가 오염될 수 있었다. | [src/build/trainer.py:28](src/build/trainer.py#L28): optimizer 생성 전 유한·비음수 검사. | [test_build.py:126](tests/test_build.py#L126): NaN·양/음의 Inf·음수 거부. 수정 전에는 +Inf 사례가 실패했다. |
| P2 | 무시·패딩 픽셀의 NaN/Inf도 CE·softmax·BCE 후 0을 곱하면 NaN을 남겼다. 보조 손실을 사용하지 않을 때도 전체 logits 합계로 만든 0이 오염됐다. | [src/train/loss.py:81](src/train/loss.py#L81), [objective.py:43](src/train/objective.py#L43), [objective.py:61](src/train/objective.py#L61): 수치 연산 전에 비활성 위치를 마스킹하고 segmentation 손실에서 보조 0을 만든다. | [test_loss.py:103](tests/test_loss.py#L103), [GDT:129](tests/test_loss.py#L129): 정상 입력과 loss·gradient 동등성, 전체 ignore, NaN/양·음 Inf, GDT label을 검사. 신규 7개가 수정 전 실패하고 수정 후 통과했다. |
| P2 | 재개 시 저장된 EMA decay가 생성자의 `[0,1)` 검사 없이 대입됐다. 손상된 메타데이터가 이후 EMA 저장값을 오염시킬 수 있었다. | [src/train/checkpoint.py:62](src/train/checkpoint.py#L62): 모델·optimizer 상태를 적용하기 전에 decay를 검사한다. | [test_trainer.py:426](tests/test_trainer.py#L426): NaN·Inf·음수·1.0 거부 및 실패 시 model·optimizer 불변 확인. |
| P2 | 설치된 TensorBoard의 전이 의존성 Werkzeug 3.1.8에서 Windows 특수 장치 경로 관련 취약점이 탐지됐다. 이 저장소 API는 해당 파일 제공 함수를 직접 호출하지 않는다. | [requirements.txt:11](requirements.txt#L11): `werkzeug>=3.1.9`, 로컬 환경도 3.1.9로 갱신. [Werkzeug 공식 공지](https://github.com/pallets/werkzeug/security/advisories/GHSA-g6x2-hccm-hh4m)는 Windows/NTFS 파일 제공 시 멈춤 조건을 설명한다. | pip-audit에서 1건 탐지 → 갱신 후 알려진 취약점 0건. |
| P3 | `max_grad_norm=+Inf`가 양수 검사를 통과해 설정 실수를 놓쳤다. | [src/train/trainer.py:37](src/train/trainer.py#L37): 유한 양수만 허용한다. | [test_trainer.py:367](tests/test_trainer.py#L367): 기존 0·음수·NaN 사례에 Inf 회귀 추가. |

## 모듈 구조와 의존성

학습 경로는 `run_train → src.run → build → data/model/train`, 추론 경로는
`run_api/backend 또는 scripts.predict → predict → model/prepare`다.
`prepare`가 RGB/SEM 변환과 resize/padding/복원 규칙을 공유하고,
`precision`이 선택 장치의 BF16/FP16 정책을 공유한다. 학습의 native-size
검증용 predictor는 builder에서 주입하므로 `train`이 `predict`에 직접
의존하지 않는다. import 경계는 `tests/test_structure.py`에서 검사한다.

| 영역과 검토 파일 | 책임 및 연결 | 판단과 검증 범위 |
| --- | --- | --- |
| `run_train.py`, `src/run.py`, `src/config.py`, `config/*.yaml` | 설정 병합·이전 설정 변환, 실행 디렉터리, 저장된 split/config 재개 | config와 resume 상호 배제, split 확인 후 모델 할당, 새 run의 config/CSV 저장 순서를 유지. `test_config`, `test_run`으로 검증. |
| `src/build/{data,model,loss,trainer}.py` | 설정에서 loader·모델·objective·AdamW·schedule·trainer 조립 | 조립 책임이 명확하다. worker/batch/augmentation 검증은 기존 보호를 유지하고 optimizer 수치 검증을 보완. `test_build`로 검증. |
| `src/data/{pairs,split,image,dataset,augment}.py`, `src/data/__init__.py` | 파일 stem pairing, split CSV·재개 일치성, palette label 읽기, crop/회전/jitter/edge mask | 이미지와 마스크의 정렬, class ID, ignore/padding, split 중복·drift 검사를 유지. `test_dataset`, `test_split`, `test_flip`, `test_sample_pipeline`로 검증. |
| `src/prepare/{spec,convert,fit}.py` | 고정 1024 입력과 metadata, SEM 통계/CLAHE, letterbox 및 native-size 복원 | 학습·API·CLI가 같은 변환을 재사용한다. 이미지와 GT의 보간 방식 분리, ROI·padding·가로세로비를 `test_preprocess`로 검증. |
| `src/model/{net,swin,output,checkpoint}.py`, `src/model/__init__.py` | 두 해상도 Swin backbone, 동결 정책, 구조화된 출력, full/base checkpoint | backbone·BatchNorm eval, state key/shape 엄격 검사, head만 재초기화. `test_model`, `test_base_compat` 통과. 단일 클래스 경로는 upstream 비교용이므로 보존. |
| `src/model/decoder/{net,block,aspp,deform}.py` | 다단계 decoder, deformable convolution/ASPP, guidance 및 deep supervision | 가중치 호환성과 gradient 연결을 확인. 근거 없는 구조·채널 수 축소는 하지 않음. tiny model forward/backward와 base checkpoint shape를 검증. |
| `src/precision.py` | 모델이 위치한 CUDA 장치에서 native BF16 확인, FP16 대체, CPU autocast 우회 | 선택 장치·fallback 테스트 유지. 실제 GPU 경로는 CPU 환경에서 실행하지 못함. |
| `src/train/{loss,objective,metrics}.py` | CE·Dice·boundary·GDT objective, mIoU/Dice 및 calibration | 비활성 픽셀의 수치 오염 수정. FP32 reduction, ignore-aware metric, 클래스별 confusion 동작을 `test_loss`, `test_metrics`로 검증. |
| `src/train/{trainer,schedule,ema,validate,checkpoint}.py` | accumulation/AMP/clip, 성공 step의 EMA·schedule, live-model 검증, atomic 저장·재개 | 잘못된 clipping/EMA 설정 거부 추가. skip된 optimizer step, 최종 저장·검증, scheduler 재개, EMA buffer, 예외 cleanup을 관련 테스트로 검증. |
| `src/predict/{model,inference,tile}.py` | run config·모델 로딩, 1024 추론, 복원, overlap 가중 합성, labels/probability 출력 | 기존 배열 재사용, tile coverage, class ID PNG, metadata 우선순위를 유지. `test_inference`, `test_inspection_scripts`로 검증. |
| `run_api.py`, `backend/{app,route,schema,codec}.py`, `backend/__init__.py` | CLI, HTTP 계약, base64/Pillow decode, 직렬 추론, PNG encode | 사용자 제어 입력의 자원 상한 3개 추가. image 오류, class/tiles 검증, chunked body, health·CLI와 색인 PNG 검증. |
| `scripts/{check_data,check_gradients,check_upstream,predict,_preview,make_sample_data}.py`, `data/` | 데이터·gradient·upstream 점검, 예측 미리보기, 합성 fixture 생성 | `check_data --batches 2` 실실행 성공. script/gradient/sample 테스트 통과. upstream 비교는 명시적 checkout을 받으며 자동 다운로드하지 않음. |
| `src/__init__.py`, `tests/conftest.py`, `tests/test_*.py`, `.gitignore`, `README.md`, `requirements.txt` | 패키지·테스트 진입점, 운영 문서, 의존성·로컬 출력 경계 | 74개 추적 Python 파일 및 설정/문서 범위를 검토. README에 의존성·새 API 한계와 감사 진입점 추가. |

주요 외부 의존성은 torch/torchvision/timm/kornia/einops(모델),
NumPy/OpenCV/Pillow(영상), OmegaConf(설정), TensorBoard/tqdm(학습 기록),
FastAPI/Pydantic/Uvicorn(HTTP), httpx2/pytest(검증)다. httpx2는 오타가 아니라
현행 Starlette TestClient가 사용하는 의존성이다.
[Starlette 공식 문서](https://www.starlette.io/testclient/)

검증 환경은 Windows, Python 3.13.15, Torch 2.14.0+cpu, torchvision 0.29.0,
NumPy 2.5.3, OpenCV 5.0.0.93, Pillow 12.3.0, FastAPI 0.141.1,
Starlette 1.7.0, Pydantic 2.13.5, pytest 9.1.1이다. requirements는 보안 하한을
설정하며 모든 전이 의존성을 고정하는 lock file은 아니다.

## 성능과 코드 품질

주요 계산 지점은 [두 해상도 backbone 실행](src/model/net.py#L67),
[deformable ASPP](src/model/decoder/aspp.py#L87),
[타일별 순차 배치 추론과 복원](src/predict/inference.py#L41)이다.
타일 연산량은 grid의 제곱으로 증가하고, native logits 메모리는
`클래스 수 × 높이 × 너비 × 4바이트`로 증가한다. API의 작업량과 배열
예산으로 사용자 입력에 따른 폭증을 제한했다. 256 MiB는 임시 배열·모델
activation까지 포함한 전체 RAM/VRAM 보장이 아니다.

동결 backbone, 배치별 tensor 전송, inference mode, tile 누적 배열 재사용,
단일 추론 세마포어는 유지했다. 모델의 두 해상도 구조와 1024 canvas를
변경하면 기본 가중치·입출력 계약에 영향을 주므로 임의로 바꾸지 않았다.
실제 GPU 병목 순위나 속도 향상 배율은 측정하지 않았다.

모듈별 책임과 import 경계는 대체로 명확하며 전역적인 구조 변경은 필요하지
않았다. 수치 검증은 소유 모듈에서 수행하고, HTTP 제한은 HTTP 경계에만
추가했다. 직접 Python 추론의 tile 입력 계약과 정상 학습/재개 형식은
유지한다. 저장은 파일별 atomic replace이며 여러 checkpoint 파일을 묶은
트랜잭션은 아니다.

## 검증과 오류 재분석

| 검사 | 결과 |
| --- | --- |
| 초기 전체 pytest | 336 passed, 652.31초. |
| 수정 후 API·전처리·추론 6개 파일 | 102 passed, 168.84초. |
| 수정 후 학습·데이터·builder 10개 파일 | 194 passed, 163.45초. 신규 16개 회귀 포함. |
| 모델·base checkpoint | 12 passed, 133.38초. 로컬 기본 checkpoint의 실제 key/shape 검사는 skip 없이 실행. |
| Swin compile 의심 재검증 | `torch.compile(backend="eager", fullgraph=True)`, shifted window, batch 2, 7×9 입력이 eager와 일치. 의심이 반증되어 소스 변경과 임시 테스트를 남기지 않음. |
| 최종 전체 pytest | **364 passed**, 실패·skip 0개, 319.63초. 초기 336개 대비 회귀 28개 추가. 경고 1개는 기존 `torch.jit.script` 폐기 예정 안내. |
| pip-audit | 설치 패키지 56개 검사. 최초 Werkzeug 1건 → 3.1.9 갱신 후 알려진 취약점 0건. |
| pip check | 의존성 충돌 없음. |
| Ruff `E9,F63,F7,F82` 및 `git diff --check` | 통과. 스타일 전체 또는 정적 보안 분석기 전체를 실행한 것으로 해석하지 않음. |
| 데이터 검사 CLI | 3개 train 이미지 중 2개 batch 검사. 유효 class pixel `[597632, 742192, 34848, 25280]`, preview/JSON 생성 성공. |
| 독립 검토와 소형 실행 | 새 P1/P2 문제 없음. 타일 타입·64패치 경계, 무시 NaN의 loss/gradient, chunked body 413을 독립 실행. README 데이터 검사 `--help` 실행과 학습 재개 경로 추적 성공. |

추가 회귀는 실패를 숨기도록 기대값을 바꾸지 않았다. 무시 픽셀 손실은
수정 전 7개 실패를 확인했고, weight decay는 4개 후보 중 +Inf만 기존
AdamW에서 허용되는 것으로 원인을 좁혔다. 활성 픽셀의 비정상 값은 여전히
학습 실패로 검출한다. 손실 경계에서의 보정이 이미 오염된 모델 내부
activation이나 가중치를 복구한다는 의미는 아니다.

첫 API 검증은 시스템 Python에 pytest가 없어 실패하여 `.venv`로 재실행했다.
병렬 실행 중 지연이 커져 후속 검증은 OMP/MKL 스레드 수를 1로 제한했다.
일부 지연된 중간 검증은 중단 후 재실행했으며 최종 통과 결과만 위 표에
기록했다. Werkzeug 설치 명령의 따옴표가 literal 인수로 전달된 오류는
`werkzeug==3.1.9` 명령으로 수정했다. 설치 중간에 실행한 pip check의
임시 `~erkzeug` 경고는 설치 종료 후 재검사에서 사라졌다.

저장소 루트의 PowerShell에서 같은 최종 검증을 실행할 수 있다.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m pytest -q --durations=10 --junitxml=run/checks/audit/pytest.xml
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_data.py --batches 2 --output run/checks/audit/data
$env:PIPAPI_PYTHON_LOCATION = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m uv tool run --from pip-audit pip-audit --path .venv/Lib/site-packages
```

JUnit, dependency audit JSON, 데이터 미리보기는 로컬 `run/checks/`에 생성되며
모델/학습 출력과 함께 Git에서 제외된다. 위 감사 명령은 검증 환경에 설치된
uv를 사용하며 일반 실행 의존성에 uv나 pip-audit를 추가하지 않았다.

## 적용 범위와 남은 검증 한계

API는 기본적으로 localhost에서 인증 없이 실행된다. 외부 서비스 운영에는
인증·접속 수·요청률 제한이 필요하다. 추론은 직렬화되지만 대기 요청의
JSON/base64 본문은 메모리에 남으므로 새 단일 요청 한도만으로 전체 서비스의
자원 소진을 방지한다고 보장할 수 없다.

CPU 환경에서 전체 자동화 테스트와 실제 기본 checkpoint shape 호환성을
확인했다. GPU BF16/FP16, CUDA deformable convolution, 전체 크기 pretrained
학습, 독립 upstream checkout과의 실제 출력 비교 및 SEM 일반화 정확도는
이번 실행의 검증 범위 밖이다. 합성 fixture의 통과는 실제 데이터의 품질이나
모델 성능을 보장하지 않는다. resume의 RNG/증강 순서 미복원은 기존에 문서화된
계약이며 변경하지 않았다.
