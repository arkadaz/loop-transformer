"""TurboQuant-style KV-cache compression: random rotation, MSE-optimal codebooks, 1-bit QJL residual.

Data-oblivious (no calibration). A key vector is normalised, rotated by a fixed random orthogonal
matrix so its coordinates are near-Gaussian, and each coordinate is rounded to the Lloyd-Max
codebook for N(0, 1) at ``bits`` bits. The rounding error is then sketched with a random Gaussian
projection and kept as one sign bit per coordinate (Quantized Johnson-Lindenstrauss), which makes
the estimated attention score ``<q, k>`` unbiased. Values use the codebook only.

References: Google Research, "TurboQuant" (2025); Zandieh, Daliri, Han, "QJL" (2024); Max (1960).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Positive half of the MSE-optimal (Lloyd-Max) codebook for a unit Gaussian.
_HALF_CODEBOOKS = {
    1: [0.7979],
    2: [0.4528, 1.5104],
    3: [0.2451, 0.7560, 1.3439, 2.1520],
    4: [0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326],
}


def codebook(bits: int) -> torch.Tensor:
    if bits not in _HALF_CODEBOOKS:
        raise ValueError(f"bits must be one of {sorted(_HALF_CODEBOOKS)}.")
    half = torch.tensor(_HALF_CODEBOOKS[bits])
    return torch.cat([-half.flip(0), half])


def pack_bits(bools: torch.Tensor) -> torch.Tensor:
    """[..., d] bool -> [..., d // 8] uint8 (little-endian within each byte)."""
    weights = (1 << torch.arange(8, device=bools.device)).to(torch.uint8)
    return (bools.reshape(*bools.shape[:-1], -1, 8).to(torch.uint8) * weights).sum(-1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, width: int) -> torch.Tensor:
    weights = (1 << torch.arange(8, device=packed.device)).to(torch.uint8)
    return (packed.unsqueeze(-1) & weights).ne(0).reshape(*packed.shape[:-1], width)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """[..., d] small ints -> [..., d * bits // 8] uint8."""
    levels = (codes.unsqueeze(-1) >> torch.arange(bits, device=codes.device)) & 1
    return pack_bits(levels.reshape(*codes.shape[:-1], -1).bool())


def unpack_codes(packed: torch.Tensor, width: int, bits: int) -> torch.Tensor:
    levels = unpack_bits(packed, width * bits).reshape(*packed.shape[:-1], width, bits).long()
    return (levels << torch.arange(bits, device=packed.device)).sum(-1)


@dataclass
class QuantizedKeys:
    codes: torch.Tensor  # uint8 [..., T, d * bits // 8], bit-packed
    norms: torch.Tensor  # fp16 [..., T]
    residual_norms: torch.Tensor  # fp16 [..., T]
    residual_signs: torch.Tensor  # uint8 [..., T, d // 8]

    def cat(self, other: "QuantizedKeys") -> "QuantizedKeys":
        return QuantizedKeys(*(torch.cat([a, b], dim=-2 if a.dim() == self.codes.dim() else -1) for a, b in zip(self.__dict__.values(), other.__dict__.values())))

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.__dict__.values())


@dataclass
class QuantizedValues:
    codes: torch.Tensor  # uint8 [..., T, d * bits // 8], bit-packed
    norms: torch.Tensor  # fp16 [..., T]

    def cat(self, other: "QuantizedValues") -> "QuantizedValues":
        return QuantizedValues(torch.cat([self.codes, other.codes], dim=-2), torch.cat([self.norms, other.norms], dim=-1))

    def nbytes(self) -> int:
        return self.codes.numel() + self.norms.numel() * 2


class TurboQuant:
    def __init__(self, head_dim: int, key_bits: int = 3, value_bits: int = 3, *, seed: int = 0, device="cpu"):
        if head_dim % 8:
            raise ValueError("head_dim must be a multiple of 8 so sign bits and codes pack into bytes.")
        generator = torch.Generator().manual_seed(seed)
        self.head_dim, self.key_bits, self.value_bits = head_dim, key_bits, value_bits
        self.rotation = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=generator))[0].to(device)
        self.sketch = torch.randn(head_dim, head_dim, generator=generator).to(device)
        self.key_book, self.value_book = codebook(key_bits).to(device), codebook(value_bits).to(device)

    def _encode(self, x: torch.Tensor, book: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
        norms = x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        z = (x.float() / norms) @ self.rotation.T * math.sqrt(self.head_dim)  # unit-variance coordinates
        codes = torch.bucketize(z, (book[1:] + book[:-1]) / 2)
        return pack_codes(codes, bits), norms.squeeze(-1).half()

    def _decode(self, codes: torch.Tensor, norms: torch.Tensor, book: torch.Tensor, bits: int) -> torch.Tensor:
        z = book[unpack_codes(codes, self.head_dim, bits)] / math.sqrt(self.head_dim)
        return (z @ self.rotation) * norms.float().unsqueeze(-1)

    def quantize_keys(self, k: torch.Tensor) -> QuantizedKeys:
        codes, norms = self._encode(k, self.key_book, self.key_bits)
        residual = k.float() - self._decode(codes, norms, self.key_book, self.key_bits)
        return QuantizedKeys(codes, norms, residual.norm(dim=-1).half(), pack_bits((residual @ self.sketch.T) > 0))

    def key_scores(self, q: torch.Tensor, keys: QuantizedKeys) -> torch.Tensor:
        """Unbiased ``q @ k^T``: dequantised keys plus the QJL estimate of ``q . residual``."""
        k_hat = self._decode(keys.codes, keys.norms, self.key_book, self.key_bits)
        signs = unpack_bits(keys.residual_signs, self.head_dim).to(q.dtype) * 2 - 1
        correction = (q @ self.sketch.T.to(q.dtype)) @ signs.transpose(-1, -2) * keys.residual_norms.to(q.dtype).unsqueeze(-2)
        return q @ k_hat.to(q.dtype).transpose(-1, -2) + correction * (math.sqrt(math.pi / 2) / self.head_dim)

    def quantize_values(self, v: torch.Tensor) -> QuantizedValues:
        return QuantizedValues(*self._encode(v, self.value_book, self.value_bits))

    def dequantize_values(self, values: QuantizedValues) -> torch.Tensor:
        return self._decode(values.codes, values.norms, self.value_book, self.value_bits)

    def bits_per_coordinate(self) -> tuple[float, float]:
        """Effective (keys, values) bits per stored coordinate, including fp16 norms and QJL sign bits."""
        return self.key_bits + 1 + 32 / self.head_dim, self.value_bits + 16 / self.head_dim
