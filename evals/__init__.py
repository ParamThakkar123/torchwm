"""Evaluation metrics for world model quality.

Provides FID, FVD, and LPIPS metrics as specified in
DIAMOND Appendix M (Alonso et al., 2024).
"""

from evals.fid import FID
from evals.fvd import FVD
from evals.lpips import LPIPS

__all__ = ["FID", "FVD", "LPIPS"]
