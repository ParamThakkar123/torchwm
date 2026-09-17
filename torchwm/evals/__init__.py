from typing import Any


def __getattr__(name: str) -> Any:
    if name == "FID":
        from torchwm.evals.fid import FID

        return FID
    if name == "FVD":
        from torchwm.evals.fvd import FVD

        return FVD
    if name == "LPIPS":
        from torchwm.evals.lpips import LPIPS

        return LPIPS
    if name == "PSNR":
        from torchwm.evals.psnr import PSNR

        return PSNR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FID", "FVD", "LPIPS", "PSNR"]
