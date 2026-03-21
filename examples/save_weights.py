# SPDX-License-Identifier: MIT
"""
Save Skala weights locally for offline use.

This script downloads weights from HuggingFace and saves them in a format
that can be loaded directly into the JAX model without needing the skala
package or internet access.
"""
# ruff: noqa: E402

import sys
from pathlib import Path

dev_path = Path(__file__).parent
sys.path.insert(0, str(dev_path))

import json
import numpy as np

from skala.functional import load_functional
from skala.functional.model import SkalaFunctional as TorchSkalaFunctional


def save_weights(output_dir: Path | str = None):
    """Save Skala weights to local files.

    Parameters
    ----------
    output_dir : Path or str, optional
        Directory to save weights. Defaults to dev/skalax/weights/
    """
    if output_dir is None:
        output_dir = dev_path / "skalax" / "weights"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading weights from HuggingFace...")
    func_torch = load_functional("skala")

    # Clean up state dict keys
    state_dict = {
        k.replace("_traced_model.", ""): v
        for k, v in func_torch.state_dict().items()
    }

    # Create fresh model and load weights to get buffers too
    print("Creating model to extract buffers...")
    torch_model = TorchSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
    )
    torch_model.load_state_dict(state_dict, strict=True)
    torch_model.double()

    # Save weights as numpy arrays
    weights_file = output_dir / "skala_weights.npz"
    print(f"Saving weights to {weights_file}...")

    weights_dict = {}
    for k, v in state_dict.items():
        # Convert to float64 numpy array
        weights_dict[k] = v.detach().cpu().numpy().astype(np.float64)

    np.savez(weights_file, **weights_dict)

    # Save buffers (Wigner 3j coefficients) from the non-local model
    buffers_file = output_dir / "skala_buffers.npz"
    print(f"Saving buffers to {buffers_file}...")

    buffers_dict = {}
    if torch_model.non_local_model is not None:
        # tp_down buffers
        for name, buf in torch_model.non_local_model.tp_down.named_buffers():
            key = f"non_local_model.tp_down.{name}"
            buffers_dict[key] = buf.detach().cpu().numpy().astype(np.float64)

        # tp_up buffers
        for name, buf in torch_model.non_local_model.tp_up.named_buffers():
            key = f"non_local_model.tp_up.{name}"
            buffers_dict[key] = buf.detach().cpu().numpy().astype(np.float64)

    np.savez(buffers_file, **buffers_dict)

    # Save model config
    config_file = output_dir / "config.json"
    print(f"Saving config to {config_file}...")

    config = {
        "lmax": 3,
        "non_local": True,
        "non_local_hidden_nf": 16,
        "radius_cutoff": 5.0,
    }
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nWeights saved to {output_dir}/")
    print(f"  - skala_weights.npz ({weights_file.stat().st_size / 1024:.1f} KB)")
    print(f"  - skala_buffers.npz ({buffers_file.stat().st_size / 1024:.1f} KB)")
    print(f"  - config.json")

    return output_dir


if __name__ == "__main__":
    save_weights()
