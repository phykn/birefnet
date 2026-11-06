import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from src.config import ROOT, load_config
from src.data.io import pairs, read_image, read_mask
from src.data.split import make
from src.prepare import fit_tensor
from src.predict import plan, weigh

OLD = Path(os.environ["BIREFNET_OLD"]) if "BIREFNET_OLD" in os.environ else None
pytestmark = pytest.mark.skipif(
    OLD is None or not (OLD / "src/prepare/spec.py").exists(),
    reason="old code unavailable",
)


def old(code):
    subprocess.run(
        [sys.executable, "-c", "import torch; torch.set_num_threads(2);\n" + code],
        cwd=OLD,
        check=True,
    )


def test_readers_preprocess_split(tmp_path):
    if not (ROOT / "data/train/image").exists():
        pytest.skip("dataset unavailable")
    ps = pairs(ROOT / "data/train/image", ROOT / "data/train/mask")
    out = tmp_path / "old.npz"
    old(f"""import numpy as np, random
from src.data.image import read_image,read_mask
from src.data.pairs import pair_files
from src.data.split import make
from src.prepare.convert import to_gray
from src.prepare.fit import fit_tensor
p={ps!r}
random.seed(0)
s=make(p,.1)
result={{'train':np.array(s['train']), 'valid':np.array(s['valid'])}}
for i,(a,b) in enumerate(p[:20]):
 result['image'+str(i)]=to_gray(read_image(a));result['mask'+str(i)]=read_mask(b)
for i,(a,b) in enumerate(p[:5]): result['tensor'+str(i)]=fit_tensor(read_image(a),is_sem=True)[0]
np.savez({str(out)!r},**result)
""")
    data = np.load(out)
    groups = make(ps, 0.1, 0)
    for key in groups:
        assert np.array_equal(data[key], groups[key])
    assert len(groups["valid"]) == 363 and len(groups["train"]) == 3267
    for i, (a, b) in enumerate(ps[:20]):
        np.testing.assert_array_equal(read_image(a), data["image" + str(i)])
        np.testing.assert_array_equal(read_mask(b), data["mask" + str(i)])
    for i, (a, b) in enumerate(ps[:5]):
        np.testing.assert_allclose(
            fit_tensor(read_image(a), is_sem=True)[0],
            data["tensor" + str(i)],
            atol=1.0e-5,
            rtol=0,
        )


def test_tiles(tmp_path):
    out = tmp_path / "tiles.pt"
    old(f"""import torch
from src.predict.tile import plan,weigh
result=[]
for h,w in [(200,200),(640,1024),(948,950)]:
 for grid in (2,3,4):
  boxes=plan(h,w,grid)
  result.append([(vars(b),torch.from_numpy(weigh(b))) for b in boxes])
torch.save(result,{str(out)!r})
""")
    data = iter(torch.load(out, weights_only=True))
    for h, w in [(200, 200), (640, 1024), (948, 950)]:
        for grid in (2, 3, 4):
            for box, (meta, weight) in zip(plan(h, w, grid), next(data)):
                assert vars(box) == meta
                np.testing.assert_array_equal(weigh(box), weight.numpy())


def test_model_output(tmp_path):
    weight = ROOT / "weight/BiRefNet-general-epoch_244.pth"
    if not weight.exists():
        pytest.skip("base unavailable")
    out = tmp_path / "model.pt"
    old(f"""import torch
from src.model.net import BiRefNet
from src.model.checkpoint import load_base,trained_state
torch.manual_seed(0)
m=BiRefNet(depth=None).eval()
load_base(m,torch.load({str(weight)!r},weights_only=True,map_location='cpu'))
x=torch.randn(1,3,64,64)
with torch.no_grad(): y=m(x).logits[-1]
torch.save({{'x':x,'y':y,'state':trained_state(m.state_dict())}},{str(out)!r})
""")
    from src.model.checkpoint import build, load_trained

    state = torch.load(out, weights_only=True)
    m = build(load_config()).eval()
    load_trained(m, state["state"])
    with torch.no_grad():
        actual = m(state["x"]).logits[-1]
    torch.testing.assert_close(actual, state["y"], atol=1.0e-4, rtol=0)
    m.train()
    assert not m.bb.training
    assert all(
        not v.training
        for v in m.modules()
        if isinstance(v, torch.nn.modules.batchnorm._BatchNorm)
    )
    assert all(
        k.startswith(("squeeze_module.", "decoder."))
        for k, v in m.named_parameters()
        if v.requires_grad
    )


def test_deep_and_sem_sample(tmp_path):
    from PIL import Image

    deep = tmp_path / "deep.tif"
    Image.fromarray(np.array([[1000, 2048, 4095]], np.uint16)).save(deep)
    paths = [str(deep)]
    sample = OLD / "samples/cross_section.png"
    if sample.exists():
        paths.append(str(sample))
    out = tmp_path / "extra.npz"
    old(
        f"from src.data.image import read_image; from src.prepare.fit import fit_tensor; import numpy as np; np.savez({str(out)!r}, **{{str(i):fit_tensor(read_image(p),is_sem=True)[0] for i,p in enumerate({paths!r})}})"
    )
    expected = np.load(out)
    for i, p in enumerate(paths):
        np.testing.assert_allclose(
            fit_tensor(read_image(p), is_sem=True)[0],
            expected[str(i)],
            atol=1.0e-5,
            rtol=0,
        )
