import torch
import torch.nn as nn

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _find_module_in_checkpoint(obj: Any) -> Optional[torch.nn.Module]:
    """Try to extract an nn.Module from common checkpoint container formats."""
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, torch.jit.ScriptModule):
        return obj
    if isinstance(obj, dict):
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


def _is_lstm_weight_key(name: str) -> bool:
    n = name.lower()
    return "weight_ih_l" in n or "weight_hh_l" in n or "bias_ih_l" in n or "bias_hh_l" in n


def _is_gru_weight_key(name: str) -> bool:
    n = name.lower()
    return "gru" in n and ("weight_ih_l" in n or "weight_hh_l" in n)


def _longest_common_prefix_dot(keys: List[str]) -> str:
    if not keys:
        return ""
    parts_list = [k.split(".") for k in keys]
    common = parts_list[0]
    for parts in parts_list[1:]:
        i = 0
        while i < len(common) and i < len(parts) and common[i] == parts[i]:
            i += 1
        common = common[:i]
    return ".".join(common)


def _find_actor_prefix(tensor_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    """
    Find a key prefix that covers the policy / actor branch only (not critic / value).
    Uses keys whose path contains actor-like tokens.
    """
    weight_keys = [k for k in tensor_dict if isinstance(tensor_dict[k], torch.Tensor)]
    actor_like: List[str] = []
    # Include shared A2C stacks (actor + critic under same prefix, e.g. a2c_network.*)
    tokens = ("actor", "policy", "a2c_actor", "actor_mlp", "a2c_network", "mu", "pi")
    for k in weight_keys:
        kl = k.lower()
        if any(t in kl for t in tokens):
            actor_like.append(k)
    if not actor_like:
        return None
    prefix = _longest_common_prefix_dot(actor_like)
    return prefix if prefix else None


def _keys_under_prefix(all_keys: List[str], prefix: str) -> List[str]:
    p = prefix + "."
    return [k for k in all_keys if k == prefix or k.startswith(p)]


def _is_critic_or_value_key(name: str) -> bool:
    """Drop value / critic heads so they are not chained into the actor MLP."""
    n = name.lower()
    return any(
        t in n
        for t in (
            "critic",
            "value_head",
            "values.",
            ".vf",
            "_vf.",
            "value.",
            "estimator",
        )
    )


class _LinSpec:
    __slots__ = ("key", "w", "b", "in_f", "out_f")

    def __init__(self, key: str, w: torch.Tensor, b: torch.Tensor) -> None:
        self.key = key
        self.w = w
        self.b = b
        self.in_f = int(w.shape[1])
        self.out_f = int(w.shape[0])


def _collect_linear_specs(
    tensor_dict: Dict[str, torch.Tensor], candidate_keys: List[str]
) -> List[_LinSpec]:
    """Collect Linear layers; skip LSTM/GRU params and critic-style keys."""
    specs: List[_LinSpec] = []
    seen_bases = set()
    for wk in sorted(set(candidate_keys), key=_natural_sort_key):
        if not wk.endswith(".weight"):
            continue
        if _is_lstm_weight_key(wk) or _is_gru_weight_key(wk):
            continue
        if _is_critic_or_value_key(wk):
            continue
        w = tensor_dict.get(wk)
        if w is None or not isinstance(w, torch.Tensor) or w.ndim != 2:
            continue
        bk = wk[:-7] + ".bias"
        b = tensor_dict.get(bk)
        if b is None or not isinstance(b, torch.Tensor):
            continue
        base = wk[: -len(".weight")]
        if base in seen_bases:
            continue
        seen_bases.add(base)
        specs.append(
            _LinSpec(
                wk,
                w.detach().cpu().float(),
                b.detach().cpu().float(),
            )
        )
    return specs


def _chain_actor_linear_specs(
    specs: List[_LinSpec],
    obs_dim: Optional[int],
    actor_out_dim: Optional[int],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Follow a valid linear chain (shared trunk -> actor head), not sorted key order.
    At a branch (same in_f, multiple layers), prefer actor_out_dim if set, else the
    output that is not a scalar value head (out_f != 1 when multiple choices).
    """
    if not specs:
        return []

    by_in: Dict[int, List[_LinSpec]] = defaultdict(list)
    all_outs = {s.out_f for s in specs}
    all_ins = {s.in_f for s in specs}
    for s in specs:
        by_in[s.in_f].append(s)

    if obs_dim is None:
        entry_dims = all_ins - all_outs
        if len(entry_dims) == 1:
            cur_in = entry_dims.pop()
        else:
            cur_in = min(all_ins)
    else:
        cur_in = obs_dim

    chain: List[_LinSpec] = []
    used_keys = set()

    while True:
        candidates = [c for c in by_in.get(cur_in, []) if c.key not in used_keys]
        if not candidates:
            break
        if len(candidates) == 1:
            pick = candidates[0]
        else:
            if actor_out_dim is not None:
                pref = [c for c in candidates if c.out_f == actor_out_dim]
                pick = pref[0] if pref else max(candidates, key=lambda c: c.out_f)
            else:
                non_scalar = [c for c in candidates if c.out_f != 1]
                pick = (
                    max(non_scalar, key=lambda c: c.out_f)
                    if non_scalar
                    else candidates[0]
                )

        chain.append(pick)
        used_keys.add(pick.key)
        cur_out = pick.out_f

        if actor_out_dim is not None and cur_out == actor_out_dim:
            break

        cur_in = cur_out

    return [(s.w, s.b) for s in chain]


class _InferredMLP(nn.Module):
    """MLP reconstructed from linear layer tensors (Tanh between layers)."""

    def __init__(self, layers: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__()
        modules: List[nn.Module] = []
        for idx, (weight, bias) in enumerate(layers):
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            modules.append(linear)
            if idx < len(layers) - 1:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ActorLSTMExport(nn.Module):
    """LSTM + head for ONNX export; single obs input, zero initial hidden/cell."""

    def __init__(self, lstm: nn.LSTM, head: Optional[nn.Module]) -> None:
        super().__init__()
        self.lstm = lstm
        self.head = head if head is not None else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        out, _ = self.lstm(obs)
        last = out[:, -1, :]
        return self.head(last)


def _parse_lstm_from_keys(
    tensor_dict: Dict[str, torch.Tensor], lstm_prefix: str
) -> Optional[nn.LSTM]:
    """
    Build nn.LSTM from keys like prefix.weight_ih_l0, weight_hh_l0, ...
    """
    p = lstm_prefix
    ih0 = p + ".weight_ih_l0"
    hh0 = p + ".weight_hh_l0"
    if ih0 not in tensor_dict or hh0 not in tensor_dict:
        return None
    w_ih = tensor_dict[ih0]
    if w_ih.ndim != 2:
        return None
    hidden_size = w_ih.shape[0] // 4
    input_size = w_ih.shape[1]
    max_layer = 0
    for k in tensor_dict:
        if k.startswith(p + ".weight_ih_l"):
            suffix = k.split("weight_ih_l", 1)[-1]
            try:
                idx = int(suffix)
                max_layer = max(max_layer, idx)
            except ValueError:
                pass
    num_layers = max_layer + 1
    lstm = nn.LSTM(
        input_size,
        hidden_size,
        num_layers,
        batch_first=True,
        bidirectional=False,
    )
    sub: Dict[str, torch.Tensor] = {}
    lp = lstm_prefix
    for k, v in tensor_dict.items():
        if k == lp:
            continue
        if k.startswith(lp + "."):
            sub[k[len(lp) + 1 :]] = v
    try:
        lstm.load_state_dict(sub, strict=False)
    except Exception:
        return None
    return lstm


def _find_lstm_prefix_under(actor_keys: List[str]) -> Optional[str]:
    for k in actor_keys:
        if ".weight_ih_l0" in k:
            return k.split(".weight_ih_l0")[0]
    return None


def _infer_actor_from_state_dict(
    state_dict: Dict[str, Any],
    actor_prefix: Optional[str] = None,
    obs_dim: Optional[int] = None,
    actor_out_dim: Optional[int] = None,
) -> Optional[nn.Module]:
    """
    Reconstruct actor: prefer LSTM + MLP head when LSTM weights exist; else MLP only.
    Only uses tensors under actor_prefix (or auto-detected), never merges critic weights.
    """
    tensor_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    if not tensor_dict:
        return None

    prefix = actor_prefix
    if prefix is None:
        prefix = _find_actor_prefix(tensor_dict)
    if prefix is None:
        print(
            "convert_pth_to_onnx: no actor-like key prefix found "
            "(expected names containing actor/policy/mu). "
            "Pass actor_prefix=... to convert_pth_to_onnx()."
        )
        return None

    all_k = list(tensor_dict.keys())
    scoped = _keys_under_prefix(all_k, prefix)
    if not scoped:
        return None

    lstm_prefix = _find_lstm_prefix_under(scoped)
    linear_keys = [k for k in scoped if not _is_lstm_weight_key(k)]

    if lstm_prefix:
        lstm = _parse_lstm_from_keys(tensor_dict, lstm_prefix)
        if lstm is not None:
            head_keys = [k for k in linear_keys if not k.startswith(lstm_prefix + ".")]
            head_specs = _collect_linear_specs(tensor_dict, head_keys)
            # Head sub-graph: infer entry dim from tensors (not global obs_dim).
            pairs_head = _chain_actor_linear_specs(head_specs, None, actor_out_dim)
            head: Optional[nn.Module]
            if pairs_head:
                head = _InferredMLP(pairs_head).net
            else:
                head = None
            print(f"Reconstructed actor with LSTM at '{lstm_prefix}' and prefix '{prefix}'.")
            return _ActorLSTMExport(lstm, head)

    specs = _collect_linear_specs(tensor_dict, scoped)
    pairs = _chain_actor_linear_specs(specs, obs_dim, actor_out_dim)
    if not pairs:
        print(f"convert_pth_to_onnx: no Linear chain under prefix '{prefix}'.")
        return None
    _extra = f", actor_out_dim={actor_out_dim}" if actor_out_dim is not None else ""
    print(
        f"Reconstructed actor MLP under prefix '{prefix}' ({len(pairs)} linear layers in chain; "
        f"critic/value keys excluded, branches resolved by shape{_extra})."
    )
    return _InferredMLP(pairs)


def _build_dummy_input(model: torch.nn.Module, input_shape: Sequence[int]) -> torch.Tensor:
    """Create dummy input: prefer first Linear; else first LSTM input size."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return torch.randn(1, module.in_features)
        if isinstance(module, nn.LSTM):
            return torch.randn(1, module.input_size)
    return torch.randn(*input_shape)


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
    actor_prefix: Optional[str] = None,
    obs_dim: Optional[int] = None,
    actor_out_dim: Optional[int] = None,
) -> Path:
    """
    Convert a .pth file to ONNX.

    For checkpoint dicts with only ``checkpoint['model']`` weights, set ``actor_prefix``
    to the submodule path of the actor (e.g. ``model.actor``) if auto-detection fails.

    For shared actor-critic trunks (e.g. ``a2c_network``), pass ``actor_out_dim`` (action
    size, e.g. 7) so the exporter follows the actor head and drops the value head.
    """
    pth_path = Path(pth_path)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    obj = torch.load(pth_path, map_location="cpu")
    model = _find_module_in_checkpoint(obj)
    if model is None and isinstance(obj, dict):
        model_state = obj.get("model")
        if isinstance(model_state, dict):
            model = _infer_actor_from_state_dict(
                model_state,
                actor_prefix=actor_prefix,
                obs_dim=obs_dim,
                actor_out_dim=actor_out_dim,
            )
    if model is None:
        details = ""
        if isinstance(obj, dict):
            details = "\n" + _describe_checkpoint_dict(obj)
        raise ValueError(
            f"Unsupported content in '{pth_path}'. "
            "Expected a serialized torch.nn.Module, or checkpoint['model'] state_dict with an actor branch. "
            f"{details}\n"
            "Tip: pass actor_prefix='your.actor.path' matching keys in state_dict."
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
    actor_prefix: Optional[str] = None,
    obs_dim: Optional[int] = None,
    actor_out_dim: Optional[int] = None,
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
                    actor_prefix=actor_prefix,
                    obs_dim=obs_dim,
                    actor_out_dim=actor_out_dim,
                )
            )
        except Exception as err:
            print(f"Failed to convert {pth_file}: {err}")
    return converted


if __name__ == "__main__":
    outputs = convert_all_pth_in_folder()
    for out in outputs:
        print(f"Exported: {out}")
