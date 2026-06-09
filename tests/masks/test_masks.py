import torch

from world_models.masks import DefaultCollator, MultiblockMaskCollator, RandomMaskCollator


def test_default_collator_returns_batch_without_masks():
    batch, masks_enc, masks_pred = DefaultCollator()([torch.tensor([1]), torch.tensor([2])])

    assert batch.shape == (2, 1)
    assert masks_enc is None
    assert masks_pred is None


def test_random_mask_collator_splits_patch_indices():
    collator = RandomMaskCollator(ratio=(0.5, 0.5), input_size=(8, 8), patch_size=4)

    batch, masks_enc, masks_pred = collator([torch.zeros(1), torch.ones(1)])

    assert batch.shape == (2, 1)
    assert masks_enc.shape == (2, 2)
    assert masks_pred.shape == (2, 2)
    assert torch.cat([masks_enc[0], masks_pred[0]]).sort().values.tolist() == [0, 1, 2, 3]


def test_multiblock_mask_collator_returns_batched_masks():
    collator = MultiblockMaskCollator(
        input_size=(16, 16),
        patch_size=4,
        enc_mask_scale=(0.5, 0.5),
        pred_mask_scale=(0.5, 0.5),
        aspect_ratio=(1.0, 1.0),
        nenc=1,
        npred=1,
        min_keep=1,
        allow_overlap=True,
    )

    batch, masks_enc, masks_pred = collator([torch.zeros(1), torch.ones(1)])

    assert batch.shape == (2, 1)
    assert masks_enc.shape[0:2] == (2, 1)
    assert masks_pred.shape[0:2] == (2, 1)
    assert masks_enc.numel() > 0
    assert masks_pred.numel() > 0
