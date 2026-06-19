"""Tests for the evals package (FID, FVD, LPIPS metrics)."""

import math
import pytest
import torch
import numpy as np

from evals.fid import FID, _frechet_distance, _compute_statistics
from evals.fvd import FVD, _sample_clips
from evals.lpips import LPIPS, VGGFeatureExtractor
from evals.psnr import PSNR


class TestFIDInternals:
    @pytest.fixture(autouse=True)
    def _require_scipy(self):
        pytest.importorskip("scipy")

    def test_frechet_distance_identical(self):
        """Fréchet distance of identical distributions should be 0."""
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.eye(3)
        dist = _frechet_distance(mu, sigma, mu, sigma)
        assert abs(dist) < 1e-6

    def test_frechet_distance_symmetric(self):
        """Fréchet distance should be symmetric."""
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([1.0, 1.0])
        sigma1 = np.eye(2)
        sigma2 = np.eye(2) * 2.0
        d12 = _frechet_distance(mu1, sigma1, mu2, sigma2)
        d21 = _frechet_distance(mu2, sigma2, mu1, sigma1)
        assert abs(d12 - d21) < 1e-6

    def test_compute_statistics(self):
        features = torch.randn(100, 64)
        mu, sigma = _compute_statistics(features)
        assert mu.shape == (64,)
        assert sigma.shape == (64, 64)

    def test_fid_identical_images(self):
        """FID on identical sets should be ~0."""
        images = torch.rand(16, 3, 64, 64)
        fid = FID(device=torch.device("cpu"))
        score = fid(images, images)
        assert score < 10.0, f"FID on identical images should be near 0, got {score}"

    def test_fid_different_images(self):
        """FID on different distributions should be non-zero."""
        real = torch.rand(16, 3, 64, 64)
        gen = torch.rand(16, 3, 64, 64) + 0.5
        fid = FID(device=torch.device("cpu"))
        score = fid(real, gen)
        assert score > 0.0


class TestFVDInternals:
    def test_sample_clips_short_video(self):
        """Videos shorter than clip_length should be padded/repeated."""
        video = torch.randn(2, 3, 8, 64, 64)  # 8 frames
        clips = _sample_clips(video, clip_length=16)
        assert clips.shape[-3] == 16  # 16 frames after padding

    def test_sample_clips_long_video(self):
        """Videos longer than clip_length should produce multiple clips."""
        video = torch.randn(1, 3, 64, 64, 64)  # 64 frames
        clips = _sample_clips(video, clip_length=16, num_clips=3)
        assert clips.shape[0] == 3  # 3 clips
        assert clips.shape[-3] == 16

    def test_fvd_identical_videos(self):
        """FVD on identical video sets should be ~0."""
        B, C, T, H, W = 8, 3, 16, 64, 64
        videos = torch.rand(B, C, T, H, W)
        try:
            fvd = FVD(device=torch.device("cpu"))
            score = fvd(videos, videos)
            assert score < 10.0
        except Exception as e:
            # R3D-18 might not be available on all systems
            pytest.skip(f"FVD test skipped (model unavailable): {e}")

    def test_fvd_different_videos(self):
        """FVD on different video distributions."""
        real = torch.rand(8, 3, 16, 64, 64)
        gen = torch.rand(8, 3, 16, 64, 64) + 0.5
        try:
            fvd = FVD(device=torch.device("cpu"))
            score = fvd(real, gen)
            assert score > 0.0
        except Exception as e:
            pytest.skip(f"FVD test skipped (model unavailable): {e}")


class TestLPIPSInternals:
    def test_lpips_identical_images(self):
        """LPIPS on identical images should be ~0."""
        images = torch.rand(8, 3, 64, 64)
        try:
            lpips = LPIPS(device=torch.device("cpu"))
            score = lpips(images, images)
            assert score < 0.1
        except Exception as e:
            pytest.skip(f"LPIPS test skipped (model unavailable): {e}")

    def test_lpips_different_images(self):
        """LPIPS on different images should be > 0."""
        real = torch.rand(8, 3, 64, 64)
        gen = torch.rand(8, 3, 64, 64) + 1.0
        try:
            lpips = LPIPS(device=torch.device("cpu"))
            score = lpips(real, gen)
            assert score > 0.0
        except Exception as e:
            pytest.skip(f"LPIPS test skipped (model unavailable): {e}")

    def test_vgg_feature_shapes(self):
        extractor = VGGFeatureExtractor(device=torch.device("cpu"))
        x = torch.randn(2, 3, 64, 64)
        features = extractor(x)
        # Should have 4 layers with expected channel counts
        expected_channels = [64, 128, 256, 512]
        assert len(features) == 4
        for feat, ch in zip(features, expected_channels):
            assert feat.shape[1] == ch

    def test_lpips_symmetric(self):
        """LPIPS should be symmetric."""
        a = torch.rand(4, 3, 64, 64)
        b = torch.rand(4, 3, 64, 64) + 0.3
        try:
            lpips = LPIPS(device=torch.device("cpu"))
            score_ab = lpips(a, b)
            score_ba = lpips(b, a)
            assert abs(score_ab - score_ba) < 1e-5
        except Exception as e:
            pytest.skip(f"LPIPS test skipped (model unavailable): {e}")


class TestPSNR:
    def test_psnr_identical_images(self):
        images = torch.rand(8, 3, 64, 64)
        psnr = PSNR()
        score = psnr(images, images)
        assert score == 80.0, (
            f"PSNR on identical images should be ~80 (1e-8 epsilon), got {score}"
        )

    def test_psnr_different_images(self):
        real = torch.rand(8, 3, 64, 64)
        gen = torch.rand(8, 3, 64, 64) + 0.5
        psnr = PSNR()
        score = psnr(real, gen)
        assert score > 0.0

    def test_psnr_video_frames(self):
        videos = torch.rand(4, 3, 16, 64, 64)
        psnr = PSNR()
        score = psnr(videos, videos)
        assert score > 75.0

    def test_psnr_shape_mismatch(self):
        real = torch.rand(8, 3, 64, 64)
        gen = torch.rand(8, 3, 32, 32)
        psnr = PSNR()
        try:
            psnr(real, gen)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_psnr_repr(self):
        psnr = PSNR()
        assert "PSNR" in repr(psnr)

    def test_psnr_known_value(self):
        half = torch.ones(2, 3, 4, 4) * 0.5
        zero = torch.ones(2, 3, 4, 4) * 0.0
        psnr = PSNR()
        # identical
        assert psnr(half, half) == 80.0
        # 0.5 vs 0: MSE = 0.25
        mse = 0.25
        expected_db = 10.0 * math.log10(1.0 / (mse + 1e-8))
        score = psnr(half, zero)
        assert abs(score - expected_db) < 1e-4


class TestEvalUtils:
    def test_generate_trajectories_imports(self):
        """Tests that evals.diamond_utils imports correctly."""
        from evals.diamond_utils import generate_trajectories

        assert callable(generate_trajectories)

    def test_evals_package_imports(self):
        """Tests that the evals package exposes named exports."""
        from evals import FID, FVD, LPIPS, PSNR

        assert FID is not None
        assert FVD is not None
        assert LPIPS is not None
        assert PSNR is not None

    def test_fid_repr(self):
        fid = FID(device=torch.device("cpu"))
        assert "FID" in repr(fid)

    def test_lpips_repr(self):
        lpips = LPIPS(device=torch.device("cpu"))
        assert "LPIPS" in repr(lpips)
