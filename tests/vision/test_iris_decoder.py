import torch

from world_models.vision.iris_decoder import IRISDecoder


def test_decode_from_indices_matches_embeddings():
    torch.manual_seed(0)

    vocab_size = 10
    embedding_dim = 8
    H = W = 4
    B = 2

    dec = IRISDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        base_channels=4,
        out_channels=3,
        frame_shape=(3, 64, 64),
    )

    # Create deterministic indices within vocab range
    flat_indices = (torch.arange(H * W) % vocab_size).unsqueeze(0).repeat(B, 1)
    indices_hw = flat_indices.view(B, H, W)

    out_from_indices = dec.decode_from_indices(indices_hw)
    assert out_from_indices.shape == (B, 3, 64, 64)

    # Compare to decode_from_embeddings using the decoder's internal embedding
    emb = dec.index_to_embedding(flat_indices)
    out_from_emb = dec.decode_from_embeddings(emb)

    assert torch.allclose(out_from_indices, out_from_emb, atol=1e-6)

    # Ensure changing an index changes the output
    indices_hw2 = indices_hw.clone()
    indices_hw2[0, 0, 0] = (indices_hw2[0, 0, 0] + 1) % vocab_size
    out_changed = dec.decode_from_indices(indices_hw2)
    # outputs for the modified sample should differ
    assert not torch.allclose(out_from_indices[0], out_changed[0])

    # Also support flat (B, HW) input
    out_from_flat = dec.decode_from_indices(flat_indices)
    assert torch.allclose(out_from_flat, out_from_indices)
