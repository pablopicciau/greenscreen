"""Scelta del dispositivo di calcolo (GPU NVIDIA con CUDA, oppure CPU).

Il calcolo per-pixel pesante (conversione del colore, chiave, proiezione del
quadro con sovracampionamento e motion blur, compositing in luce lineare)
usa PyTorch: sulla GPU è molte volte più veloce e permette algoritmi più
accurati a parità di tempo. Senza GPU lo stesso codice gira sulla CPU.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=None)
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def get_device(preference: str = "auto"):
    """Ritorna il ``torch.device`` da usare.

    ``preference``: ``"auto"`` (GPU se disponibile), ``"cuda"`` (errore se
    manca) oppure ``"cpu"``.
    """
    import torch

    pref = (preference or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if pref in ("cuda", "gpu"):
        raise RuntimeError(
            "GPU CUDA non disponibile: installa PyTorch con CUDA "
            "(vedi README) oppure usa --device cpu."
        )
    return torch.device("cpu")


def describe(device) -> str:
    """Descrizione leggibile del dispositivo (per i messaggi)."""
    import torch

    if device.type == "cuda":
        return f"GPU {torch.cuda.get_device_name(device)}"
    return "CPU"
