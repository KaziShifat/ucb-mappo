"""Export trained actors to ONNX for the robots.

    python hardware/export_onnx.py --ckpt logs/coverage/warehouse/n4/ucb/seed1/actors.pt \
        --out hardware/models/coverage_n4

Writes robot1.onnx ... robotN.onnx (each robot has its own actor). The exported graph is
one forward pass with dropout OFF, exactly as run on the Raspberry Pi 4. Input: 38 values
(float32, shape [1, 38]). Output: (v [m/s], w [rad/s]), already scaled to the velocity limits.
The exported file is checked against PyTorch with ONNX Runtime before it is saved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tb4_multi_nav"))
from tb4_multi_nav.agents.ppo import V_MAX, W_MAX, Actor  # noqa: E402


class Deployed(torch.nn.Module):
    def __init__(self, actor: Actor):
        super().__init__()
        self.actor = actor.eval()
        self.register_buffer("scale", torch.tensor([V_MAX, W_MAX]))

    def forward(self, obs):
        mean, _ = self.actor(obs, 1)
        return mean * self.scale


def export_actor(actor: Actor, path: Path) -> float:
    model = Deployed(actor).eval()
    dummy = torch.zeros(1, 38)
    torch.onnx.export(model, dummy, str(path), input_names=["obs"], output_names=["cmd"],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).uniform(0, 1, (8, 38)).astype(np.float32)
    model.eval()                      # torch.onnx.export restores train mode; dropout must stay off
    ref = model(torch.as_tensor(x)).detach().numpy()
    out = np.concatenate([sess.run(None, {"obs": x[i:i + 1]})[0] for i in range(8)])
    return float(np.abs(ref - out).max())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, sd in enumerate(ck["actors"], start=1):
        actor = Actor(ck["obs_dim"], cfg["hidden"], cfg["dropout_p"] if ck["ucb"] else 0.0)
        actor.load_state_dict(sd)
        err = export_actor(actor, out / f"robot{k}.onnx")
        print(f"robot{k}.onnx written, max difference to PyTorch {err:.2e}")


if __name__ == "__main__":
    main()
