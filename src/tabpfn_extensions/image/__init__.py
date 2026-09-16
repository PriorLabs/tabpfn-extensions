"""Image module for tabpfn_extensions package."""

from ._embeddings import GatedEncoderError
from ._preprocessing import image_to_bytes

__all__ = ["GatedEncoderError", "image_to_bytes"]
