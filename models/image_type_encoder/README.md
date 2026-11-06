# DINOv2 image-type encoder

`src/image_detection/image_type_encoder.py` provides `ImageTypeEncoder`, an
independent, pretrained feature extractor for a future image-type router. It has
no classifier and is not registered in production inference or fusion.

The backbone is Meta/Facebook **DINOv2 ViT-B/14**, loaded with Transformers'
`Dinov2Model` from the official [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base)
repository. This is the base model with 14-pixel patches, 12 layers, 12 attention
heads, 768-dimensional features, and 86,580,480 frozen parameters. It contains no
fine-tuned head. The original implementation is
[`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2).

Both weights and preprocessing are pinned to revision
`f9e44c814b77203eaa57a6bdbbd535f21ede1415`. The 346,345,912-byte
`model.safetensors` file has SHA-256
`d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`,
matching the official repository's LFS metadata. Checkpoint/config downloads stay
in `models/image_type_encoder/.cache/`, already excluded by the project's cache
ignore rule. Other checkpoints and caches are unused.

## Usage

From the project root, with the existing Python dependencies installed:

```python
from src.image_detection.image_type_encoder import ImageTypeEncoder

encoder = ImageTypeEncoder(device="cpu", local_files_only=True)
features = encoder.extract_features(
    ["tests/sample_image_real.jpg", "tests/sample_image_fake.jpg"],
    batch_size=2,
)
assert features.shape == (2, 768)
numpy_features = features.numpy()
```

Omit `local_files_only=True` on a fresh machine to allow the pinned model/config
download. Offline mode propagates missing-cache errors without an online retry.
Model loading happens only when constructing an encoder; importing the module
does not download or instantiate a model. The existing package initializer
eagerly imports detector modules, so standalone tools can avoid that initializer:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src/image_detection").resolve()))
from image_type_encoder import ImageTypeEncoder
```

The module itself imports no detection code. Keep one encoder instance and reuse
it. `device=None` or `"auto"` chooses CUDA if available, otherwise CPU;
`"cuda"`/`"cuda:0"` explicitly requires CUDA. The backbone and pixels use the
selected device; returned features always use CPU float32 for convenient storage
and later classifier training. Move features to a future head's device as needed.
No gradients are recorded for DINOv2, and extraction restores frozen evaluation
mode if a caller changes the exposed backbone's flags.

## Input and feature contract

- Input: one image or a nonempty sequence of paths, PIL images, or uint8 NumPy
  arrays. NumPy arrays use RGB by default; pass `color_order="BGR"` for OpenCV.
  Grayscale arrays and grayscale/RGBA PIL images are converted to RGB. Float
  arrays, tensor inputs, and ambiguous array channel counts are rejected.
- Preprocessing: apply EXIF orientation, convert to RGB, bicubic resize the short
  edge to 256, center crop to 224 × 224, scale by 1/255, and normalize with
  ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
  `preprocess(...)` exposes the resulting `(N, 3, 224, 224)` CPU tensor.
- Features: final layer-normalized CLS token, shape `(N, 768)`, input order
  preserved, detached float32 CPU tensor. A single image returns `(1, 768)`.
  `normalize=True` optionally applies L2 normalization. The default keeps the
  raw DINOv2 representation. These values are not scores or class probabilities.
- Extraction is chunked by `batch_size` (default 16). This bounds inference
  memory; the returned feature matrix still contains all requested embeddings.
  No face crop, quality gate, random augmentation, training, or routing occurs.

## Verification

Offline unit tests use a small backbone test double and download nothing:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_image_type_encoder.py -v
```

Real pretrained inference uses the four existing local face images
`sample_image_real.jpg`, `sample_image_fake.jpg`, `real.jpg`, and
`face-merge-simple-3.jpeg` in `tests/`. It verifies finite 768-dimensional outputs,
batch consistency, optional L2 normalization, frozen parameters, no detector
imports, checkpoint SHA-256, and CPU/CUDA agreement when CUDA is available:

```powershell
.\.venv\Scripts\python.exe scripts\verify_image_type_encoder.py
```

Add `--download` on a fresh machine to fetch only the model and preprocessing
config. Supply local image paths as positional arguments to use other existing
samples. Reports and `.npy` embeddings go to `outputs/diagnostics/dinov2/` by
default; `--output-dir` can separate runs. The verifier trains nothing and never
downloads datasets.

The project's `.venv` has CPU-only PyTorch. A separate, ignored
`experiments/.venv-dinov2-gpu` environment is used for CUDA verification without
changing that installation:

```powershell
.\experiments\.venv-dinov2-gpu\Scripts\python.exe scripts\verify_image_type_encoder.py --require-cuda --output-dir outputs\diagnostics\dinov2\gpu
```

`--require-cuda` fails if CUDA cannot actually run, rather than treating a skipped
GPU check as success. Verification does not measure routing/classification
accuracy; no routing head or label mapping exists yet.

To recreate the isolated GPU test environment used here (Python 3.14, an NVIDIA
driver compatible with CUDA 13.0), run:

```powershell
.\.venv\Scripts\python.exe -m venv experiments\.venv-dinov2-gpu
.\experiments\.venv-dinov2-gpu\Scripts\python.exe -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.\experiments\.venv-dinov2-gpu\Scripts\python.exe -m pip install transformers==5.22.1 huggingface-hub==2.4.0 safetensors==0.8.0
```

This environment needs only the encoder dependencies; the detector stack is not
required by the standalone verifier. The main project's requirements and Python
environment remain separate.

### Verified results (2026-11-05)

- All 10 offline contract tests passed.
- All four existing face samples passed pretrained inference in the original
  CPU-only environment (`torch 2.15.0+cpu`) and on CPU/CUDA in the separate GPU
  environment (`torch 2.14.1+cu130`, NVIDIA GeForce RTX 3050 6GB Laptop GPU).
- Both devices returned finite `(4, 768)` float32 embeddings, with zero trainable
  parameters. Checkpoint SHA-256, batch consistency, and L2 normalization passed.
- CPU/CUDA maximum absolute difference: `1.8209219e-05`; minimum cosine
  similarity: `0.9999998808` on these samples.
- Reports and sample embeddings: `outputs/diagnostics/dinov2/verification.json`
  and `outputs/diagnostics/dinov2/gpu/verification.json`, with corresponding
  `embeddings_*.npy` files alongside each report. No classification head was
  created or trained.
