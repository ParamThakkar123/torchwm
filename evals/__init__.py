"""Evaluation metrics for world model quality.

Provides FID, FVD, LPIPS, and PSNR metrics as specified in
DIAMOND Appendix M (Alonso et al., 2024).
"""

from evals.fid import FID
from evals.fvd import FVD
from evals.lpips import LPIPS
from evals.psnr import PSNR

__all__ = ["FID", "FVD", "LPIPS", "PSNR"]
