import pytest
import torch
from world_models.models import create_genie_small
from world_models.vision import create_video_tokenizer
from world_models.models.latent_action_model import create_latent_action_model


class TestVideoTokenizer:
    def test_initialization(self):
        tokenizer = create_video_tokenizer(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            decoder_dim=512,
            encoder_depth=4,
            decoder_depth=8,
            vocab_size=1024,
            embedding_dim=32,
        )
        assert tokenizer is not None

    def test_forward_pass(self):
        tokenizer = create_video_tokenizer(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            decoder_dim=512,
            encoder_depth=4,
            decoder_depth=8,
            vocab_size=1024,
            embedding_dim=32,
        )
        tokenizer.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            recon, indices, loss_dict = tokenizer(x)

        assert recon.shape == (B, C, T, H, W)
        assert indices.shape == (B, T, H // 4, W // 4)
        assert "recon_loss" in loss_dict
        assert "vq_loss" in loss_dict

    def test_encode_decode(self):
        tokenizer = create_video_tokenizer(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            decoder_dim=512,
            encoder_depth=4,
            decoder_depth=8,
            vocab_size=1024,
            embedding_dim=32,
        )
        tokenizer.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            z_q, indices, vq_loss = tokenizer.encode(x)
            recon = tokenizer.decode(z_q)

        assert z_q.shape[0] == B
        assert indices.shape[0] == B
        assert recon.shape == (B, C, T, H, W)


class TestLatentActionModel:
    def test_initialization(self):
        lam = create_latent_action_model(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            encoder_depth=4,
            num_heads=8,
            vocab_size=8,
            embedding_dim=32,
        )
        assert lam is not None

    def test_encode(self):
        lam = create_latent_action_model(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            encoder_depth=4,
            num_heads=8,
            vocab_size=8,
            embedding_dim=32,
        )
        lam.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x_prev = torch.randn(B, C, T, H, W)
        x_next = torch.randn(B, C, H, W)

        with torch.no_grad():
            latent_actions, z_q = lam.encode(x_prev, x_next)

        assert latent_actions.shape == (B, T)
        assert z_q.shape[0] == B


class TestGenie:
    def test_initialization(self):
        model = create_genie_small(num_frames=8, image_size=32)
        assert model is not None

    def test_get_num_parameters(self):
        model = create_genie_small(num_frames=8, image_size=32)
        num_params = model.get_num_parameters()
        assert num_params > 0
        print(f"Model parameters: {num_params:,}")

    def test_forward_pass(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            outputs = model(x, mask_prob=0.5)

        assert "reconstructed_video" in outputs
        assert "video_indices" in outputs
        assert "latent_actions" in outputs
        assert "tokenizer_loss" in outputs

        assert outputs["reconstructed_video"].shape == (B, C, T, H, W)
        assert outputs["latent_actions"].shape[0] == B

    def test_different_batch_sizes(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 8, 32, 32)
            with torch.no_grad():
                outputs = model(x, mask_prob=0.5)
            assert outputs["reconstructed_video"].shape[0] == batch_size

    def test_different_frame_counts(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.eval()

        x = torch.randn(2, 3, 4, 32, 32)
        with torch.no_grad():
            outputs = model(x, mask_prob=0.5)
        assert outputs["reconstructed_video"].shape == (2, 3, 4, 32, 32)


class TestGenieTraining:
    def test_training_step(self):
        from world_models.training.train_genie import create_genie_trainer, GenieConfig

        config = GenieConfig()
        config.max_steps = 1

        trainer, model = create_genie_trainer(config)

        B, C, T, H, W = 2, 3, 8, 32, 32
        batch = torch.randn(B, C, T, H, W)

        losses = trainer.train_step(batch)
        assert "total_loss" in losses
        assert losses["total_loss"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
