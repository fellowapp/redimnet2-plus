"""Encoder, bridge, classifier, and model assembly."""

from asv.models.assemble import ASVModel, build_model
from asv.models.bridge import IdentityBridge, LinearBridge
from asv.models.classifier import IdentityClassifier, LinearClassifier, SphereFace2Head
from asv.models.encoder import ConvWaveformEncoder, IdentityEncoder
from asv.models.redimnet2 import ReDimNet2Encoder

__all__ = [
    "ASVModel",
    "ConvWaveformEncoder",
    "IdentityBridge",
    "IdentityClassifier",
    "IdentityEncoder",
    "LinearBridge",
    "LinearClassifier",
    "SphereFace2Head",
    "ReDimNet2Encoder",
    "build_model",
]
