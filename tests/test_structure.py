import subprocess
import sys
from pathlib import Path


def test_model_output_import_stays_light_and_birefnet_export_is_compatible():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "from src.model.output import Output\n"
        "assert Output.__name__ == 'Output'\n"
        "assert 'src.model.net' not in sys.modules\n"
        "assert 'src.model.swin' not in sys.modules\n"
        "assert 'torchvision' not in sys.modules\n"
        "import src.model as model\n"
        "assert 'Output' in model.__all__\n"
        "assert 'BiRefNet' in model.__all__\n"
        "from src.model import BiRefNet\n"
        "assert BiRefNet.__name__ == 'BiRefNet'\n"
        "assert 'src.model.net' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


def test_training_import_does_not_import_prediction_or_cli():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "from src.train.trainer import Trainer\n"
        "assert not any(name.startswith('src.predict') for name in sys.modules)\n"
        "assert 'run_train' not in sys.modules\n"
        "assert 'run_api' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


def test_model_checkpoint_import_does_not_depend_on_builders_or_training():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "from src.model.checkpoint import read_checkpoint, pack_model\n"
        "for prefix in ('src.build', 'src.train', 'src.predict', 'backend'):\n"
        "    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)\n"
        "assert 'src.model.net' not in sys.modules\n"
        "assert 'torchvision' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
