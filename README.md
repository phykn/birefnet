# BiRefNet SEM segmentation

SEM 이미지를 여러 클래스로 분할하기 위한 BiRefNet 미세조정 도구입니다. Swin-L 백본을 고정하고 squeeze 모듈과 디코더만 학습합니다. CUDA, Apple GPU, CPU를 자동으로 선택합니다.

기본 학습 crop은 512입니다. 깊이 입력과 추가 손실·교사는 선택 기능이며 기본적으로 꺼져 있습니다. 장시간 비교 학습을 통한 정확도 평가는 아직 하지 않았습니다.

## 설치

Python 3.10 이상과 PyTorch 2.10 이상을 사용합니다. 기존 `.venv`가 있으면 그대로 사용합니다. 처음 설치할 때만 `uv venv .venv`로 환경을 만듭니다.

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

기반 가중치 `BiRefNet-general-epoch_244.pth`를 `weight/`에 둡니다. 실행 중 자동 다운로드는 하지 않습니다.

## 데이터

`data/train/image/`와 `data/train/mask/`에 확장자를 제외한 이름이 같은 이미지와 마스크를 둡니다. 마스크 값은 클래스 번호(기본 0–3)이며 255는 무시합니다. 이미지는 회색으로 읽고 16비트 입력은 이미지 자체의 최솟값과 최댓값으로 8비트 범위에 맞춥니다.

```bash
.venv/bin/python scripts/01_dataset.py
.venv/bin/python scripts/02_augment.py
```

## 학습

```bash
.venv/bin/python run_train.py
.venv/bin/python run_train.py --config experiments/base_512.yaml
.venv/bin/python run_train.py --resume run/<id>/weights/last.train.pth
```

`--config`에는 바꿀 키만 적습니다. 모르는 키, 잘못된 형식·범위는 모델 생성 전에 오류를 냅니다. 설정 안의 상대 경로는 프로젝트 폴더 기준이고 명령줄 경로는 현재 폴더 기준입니다. `--resume`과 `--config`는 함께 사용할 수 없습니다.

각 실행은 설정, 학습·검증 분할, TensorBoard 로그와 가중치를 `run/<id>/`에 남깁니다. 재개 시 모델, EMA, 최적화 상태와 학습률은 이어지지만 난수 상태와 데이터 순서는 이어지지 않습니다.

```bash
.venv/bin/tensorboard --logdir run
```

## 선택 기능

설정 파일에서 아래 값을 바꾸면 각각 켜집니다. 0 또는 false이면 꺼집니다. 표의 켤 때 값은 이전 레시피의 값이며, 새 기본값보다 낫다고 검증한 값은 아닙니다.

| 기능 | 설정 | 기본값 | 켤 때 예시 |
|---|---|---|---|
| 깊이 입력 | `model.depth` | `false` | `true` |
| 경계 띠 손실 | `loss.boundary` | `0.0` | `0.5` |
| 엣지 유도 손실 | `loss.aux` | `0.0` | `1.0` |
| 라벨 흐림 | `loss.label_blur` | `0.0` | `1.0` |
| EMA 교사 | `loss.teacher` | `0.0` | `0.3` |

깊이 입력은 `model.depth_weight`의 로컬 Depth Anything V2 폴더가 필요합니다(기본 `weight/Depth-Anything-V2-Small-hf`). 경계 폭은 `loss.boundary_width`(기본 3), 교사 시작은 `loss.teacher_start`(기본 1000스텝)로 설정합니다. 교사는 시작 뒤 같은 스텝 수에 걸쳐 설정 비율까지 증가하며, 학생과 고정된 백본·깊이 특징을 공유합니다.

`experiments/o3_*.yaml`에는 각 기능 하나만 켠 3000스텝 비교 설정이 있습니다. `experiments/old_all.yaml`은 이전 커밋 `750d7d9`의 코드에만 사용합니다. 현재 코드의 기본 실행에는 사용하지 않습니다.

## 예측

```bash
.venv/bin/python scripts/03_predict.py --weight run/<id>/weights/best_miou.pth --image samples/particles.png --tiles 1 3
```

`--mask PATH`로 정답을, `--class-id 1`로 특정 클래스 확률을 함께 봅니다. `--device cpu`로 장치를 지정할 수 있습니다. 예측은 원본 크기로 복원하며 화면에 표시하고 파일은 쓰지 않습니다.

## 채점

```bash
.venv/bin/python scripts/04_evaluate.py --weight run/<id>/weights/best_miou.pth --images data/test/image --masks data/test/mask
```

쌍 수, mIoU, 클래스별 IoU, 픽셀 정확도를 출력합니다. 점수 범위는 0–1입니다. 비교 실험 중에는 test 세트로 설정을 선택하지 않습니다.

## 테스트

```bash
.venv/bin/python -m pytest -q
```

기반 가중치·이전 코드·Apple GPU가 필요한 검사는 해당 자원이 없으면 건너뜁니다. `tests/test_old.py`는 이전 구현과의 비교를 모았습니다. 전환 뒤에는 `BIREFNET_OLD`로 이전 코드 위치를 지정할 수 있습니다.

## 이전 버전에서 바뀐 것

- 체크포인트는 `birefnet-multiclass-v3`이며 전체 설정을 포함합니다. 예측에는 체크포인트와 기반 가중치만 필요합니다. v2 체크포인트는 커밋 `750d7d9`의 코드로 읽습니다.
- `best_miou.pth`와 `last.pth`는 EMA 가중치입니다. `last.train.pth`에는 학습 재개 상태를 담습니다. 별도 `last_ema.pth`는 만들지 않습니다.
- `train.crop`을 추가했습니다. 기본값은 512이고, 64–1024 사이의 32 배수로 설정할 수 있습니다. 예측 입력은 1024로 유지합니다.
- `scripts/04_evaluate.py`를 추가하고 `03_predict.py --config`를 제거했습니다.
- `data.sem`, `data.ignore_index`, `model.channels`, `train.grad_checkpoint`, 로더의 `prefetch_factor`, `persistent_workers`, `pin_memory`, `drop_last` 설정을 제거했습니다.
- 기준 레시피는 CE + Dice + EMA입니다. 깊이 입력, 경계 띠 손실, 엣지 유도 손실, 라벨 흐림, EMA 교사를 설정으로 켤 수 있습니다. 기본값은 모두 끔입니다. 경계 띠는 이미지 테두리·무시 영역·잘린 창의 경계를 제외합니다.
