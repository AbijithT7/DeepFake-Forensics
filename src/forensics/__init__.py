"""
Facial Manipulation Forensics Package.

Provides computer-vision and signal-processing forensic analyzers for:
- Face swap boundary blending & Error Level Analysis (ELA)
- Facial feature alteration & bilateral symmetry
- Geometric facial distortion & edge coherence
- Skin texture smoothness & high-frequency residual
- Face-to-body color & noise consistency
"""

from src.forensics.facial_manipulation import (
    FacialManipulationForensics,
    ForensicAnalysisResult,
)

__all__ = [
    "FacialManipulationForensics",
    "ForensicAnalysisResult",
]
