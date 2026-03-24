import torch

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _find_module_in_checkpoint(obj: Any) -> Optional[torch.nn.Module]:
    """Try to extract an nn.Module from common checkpoint container formats."""
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, torch.jit.ScriptModule):
        return obj
    if isinstance(obj, dict):
        # Common keys used by RL/training checkpoints.
        for key in ("model", "policy", "actor", "module", "net"):
            value = obj.get(key)
            if isinstance(value, (torch.nn.Module, torch.jit.ScriptModule)):
                return value
    return None


def _natural_sort_key(text: str) -> List[Union[int, str]]:
    chunks: List[Union[int, str]] = []
    current = ""
    is_digit = False
    for ch in text:
        if ch.isdigit():
            if not is_digit and current:
                chunks.append(current)
                current = ""
            current += ch
            is_digit = True
        else:
            if is_digit and current:
                chunks.append(int(current))
                current = ""
            current += ch
            is_digit = False
    if current:
        chunks.append(int(current) if is_digit else current)
    return chunks


class _InferredMLP(torch.nn.Module):
    """Minimal MLP reconstructed from linear layer tensors."""

    def __init__(self, layers: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__()
        modules: List[torch.nn.Module] = []
        for idx, (weight, bias) in enumerate(layers):
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            modules.append(linear)
            if idx < len(layers) - 1:
                # Tanh is common for RL actor MLPs and is ONNX-friendly.
                modules.append(torch.nn.Tanh())
        self.net = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_dummy_input(model: torch.nn.Module, input_shape: Sequence[int]) -> torch.Tensor:
    """Create a sensible dummy input for ONNX export."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return torch.randn(1, module.in_features)
    return torch.randn(*input_shape)


def _infer_model_from_state_dict(state_dict: Dict[str, Any]) -> Optional[torch.nn.Module]:
    """Try to reconstruct a basic actor/policy MLP from checkpoint state_dict."""
    if not state_dict:
        return None

    tensor_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    if not tensor_dict:
        return None

    weight_keys = [k for k, v in tensor_dict.items() if k.endswith(".weight") and v.ndim == 2]
    if not weight_keys:
        return None

    priority_tokens = ("actor", "policy", "mu", "action")
    selected = []
    for token in priority_tokens:
        candidate = [k for k in weight_keys if token in k.lower()]
        if candidate:
            selected = candidate
            break
    if not selected:
        selected = weight_keys

    selected = sorted(selected, key=_natural_sort_key)
    layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for wk in selected:
        bk = wk[:-7] + ".bias"
        if bk not in tensor_dict:
            continue
        w = tensor_dict[wk].detach().cpu().float()
        b = tensor_dict[bk].detach().cpu().float()
        layers.append((w, b))

    if not layers:
        return None
    return _InferredMLP(layers)


def _describe_checkpoint_dict(obj: Dict[str, Any]) -> str:
    keys = sorted(obj.keys())
    lines = [f"Checkpoint keys: {keys}"]
    for key in keys:
        value = obj[key]
        if isinstance(value, torch.nn.Module):
            lines.append(f"  - {key}: nn.Module ({type(value).__name__})")
        elif isinstance(value, dict):
            lines.append(f"  - {key}: dict(len={len(value)})")
        elif isinstance(value, (list, tuple)):
            lines.append(f"  - {key}: {type(value).__name__}(len={len(value)})")
        else:
            lines.append(f"  - {key}: {type(value).__name__}")
    return "\n".join(lines)


def convert_pth_to_onnx(
    pth_path: Union[str, Path],
    onnx_path: Union[str, Path],
    input_shape: Sequence[int] = (1, 3, 224, 224),
    opset_version: int = 17,
) -> Path:
    """
    Convert a .pth file to ONNX and write it to disk.

    Notes:
    - This works when the .pth contains a serialized torch.nn.Module or TorchScript module.
    - If the file contains only a state_dict, architecture information is missing and export
      cannot be done without rebuilding the model first.
    """
    pth_path = Path(pth_path)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    obj = torch.load(pth_path, map_location="cpu")
    model = _find_module_in_checkpoint(obj)
    if model is None and isinstance(obj, dict):
        model_state = obj.get("model")
        if isinstance(model_state, dict):
            model = _infer_model_from_state_dict(model_state)
            if model is not None:
                print("Reconstructed model from checkpoint['model'] state_dict.")
    if model is None:
        details = ""
        if isinstance(obj, dict):
            details = "\n" + _describe_checkpoint_dict(obj)
        raise ValueError(
            f"Unsupported content in '{pth_path}'. "
            "Expected a serialized torch.nn.Module or TorchScript module, "
            "or one under checkpoint keys like model/policy/actor."
            f"{details}\nIf this file only contains state_dict weights, "
            "you must recreate the model architecture and load_state_dict() first."
        )

    model.eval()
    print(f"\nModel structure for: {pth_path}")
    print(model)
    dummy_input = _build_dummy_input(model=model, input_shape=input_shape)
    print(f"Using dummy input shape: {tuple(dummy_input.shape)}")
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    return onnx_path


def convert_all_pth_in_folder(
    pth_dir: Union[str, Path] = "checkpoints/pth",
    onnx_dir: Union[str, Path] = "checkpoints/onnx",
    input_shape: Sequence[int] = (1, 3, 224, 224),
    opset_version: int = 17,
) -> List[Path]:
    """Convert every .pth file in pth_dir into .onnx files in onnx_dir."""
    pth_dir = Path(pth_dir)
    onnx_dir = Path(onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    converted: List[Path] = []
    for pth_file in sorted(pth_dir.glob("*.pth")):
        onnx_path = onnx_dir / f"{pth_file.stem}.onnx"
        print(f"\nConverting: {pth_file} -> {onnx_path}")
        try:
            converted.append(
                convert_pth_to_onnx(
                    pth_path=pth_file,
                    onnx_path=onnx_path,
                    input_shape=input_shape,
                    opset_version=opset_version,
                )
            )
        except Exception as err:
            print(f"Failed to convert {pth_file}: {err}")
    return converted


if __name__ == "__main__":
    outputs = convert_all_pth_in_folder()
    for out in outputs:
        print(f"Exported: {out}")

