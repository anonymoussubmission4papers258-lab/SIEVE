# Third-party components

The learned backbone uses DINOv2 via Hugging Face Transformers. DINOv2 is an
external Meta AI model (Apache-2.0); Transformers is an external Hugging Face
library (Apache-2.0). NumPy, SciPy, PyTorch, scikit-image, scikit-learn, Pillow,
h5py and safetensors retain their respective licenses. Install dependencies
from their official distributions; no vendored environments are included.

Released learned checkpoints contain fine-tuned DINOv2 parameters. The upstream
DINOv2 license and notices are included in `licenses/`. Upstream author names in
those required notices credit third-party software, not this submission's authors.
Raw third-party datasets and cached teacher features are not redistributed.
