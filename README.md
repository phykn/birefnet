# BiRefNet SEM Segmentation

배터리 SEM 이미지를 픽셀 단위로 분할하는 **BiRefNet 다중 클래스 파인튜닝** 프로젝트입니다.
기본 4개 클래스의 학습, 타일 추론, FastAPI 서버를 제공합니다.

## 시작하기

저장소 루트에서 실행합니다.

```bash
pip install -r requirements.txt
```

1. [사전학습 가중치](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth)를 `weight/BiRefNet-general-epoch_244.pth`에 저장합니다.
2. 같은 파일명 stem의 이미지와 마스크를 `data/image/`, `data/mask/`에 넣습니다.
3. `config/train.yaml`에서 데이터 경로·해상도·학습 설정을 조정합니다. 클래스 수는 `config/model.yaml`에서 설정합니다.

마스크는 **클래스 ID 0–3을 담은 단일 채널 또는 팔레트 이미지**여야 합니다. RGB 마스크는 지원하지 않으며, `255`는 무시합니다.
포함된 [합성 샘플 4쌍](data/README.md)은 실행 확인용입니다. 실제 학습·평가에는 별도 SEM 데이터가 필요합니다.

## 학습

```bash
python run_train.py --config config/train.yaml
python run_train.py --resume run/<run-id>/weights/last.train.pth
```

기본 `decoder` 모드는 백본을 동결하고 squeeze·decoder를 학습합니다.
`train.mode: partial`은 마지막 백본 단계도, `full`은 전체 모델을 학습합니다. CUDA가 있으면 자동 사용합니다.

결과는 `run/<run-id>/`에 저장됩니다. `best_miou.pth`는 검증 mIoU 기준 최적 모델,
`last.pth`는 마지막 모델, `last.train.pth`는 재개용 상태입니다.
같은 시편에서 나온 이미지·크롭은 같은 분할에 두세요. 자동 분할은 이미지 단위입니다.

## 데이터·모델 점검

```bash
# 실제 로더의 입력, 증강, 마스크, 유효 영역 확인
python scripts/check_data.py --size 256 --batches 2

# 학습 대상의 그래디언트 연결과 유한값 확인
python scripts/check_gradients.py --size 64

# 원본 크기 예측: 클래스 ID PNG, 오버레이, 비교 이미지 저장
python scripts/predict.py --weight run/<run-id>/weights/best_miou.pth --image data/image/sample_01.png
```

점검 결과는 `run/checks/`에 저장됩니다. 각 스크립트의 `--help`에서 옵션을 확인하세요.
예측에 `--mask`를 주면 정답을 함께 표시하고, `--tiles 1 3`으로 타일 추론,
`--class-id 3`으로 특정 클래스 확률 PNG를 저장할 수 있습니다.
예측은 저장된 전처리를 재사용합니다. `labels.png`의 팔레트 인덱스가 클래스 ID입니다.

## API

```bash
python run_api.py --host 127.0.0.1 --port 8000 --weight run/<run-id>/weights/best_miou.pth
```

API 문서는 `http://127.0.0.1:8000/docs`에서 확인합니다. `POST /predict`에 `base64_str`을 전달하면 클래스 ID PNG를 반환합니다.
GPU는 `--device cuda`, 기본과 다른 클래스 설정은 `--config run/<run-id>/config.yaml`을 지정합니다.

## 개발

`src/`는 모델·데이터·학습·추론, `backend/`는 HTTP API, `scripts/`는 실행 점검 도구입니다.

```bash
python -m pytest -q
```

[원본 BiRefNet](https://github.com/ZhengPeng7/BiRefNet) 기반입니다. 현재 체크포인트는 전체 모델 형식이며, 이전 LoRA·이진 분할 체크포인트와 호환되지 않습니다.
