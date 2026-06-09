import torch

from world_models.masks import DefaultCollator, MultiblockMaskCollator, RandomMaskCollator


def _assert_mask_collection(mask_collection, batch_size, num_masks):
    if isinstance(mask_collection, torch.Tensor):
        assert mask_collection.shape[0:2] == (batch_size, num_masks)
        assert mask_collection.numel() > 0
        return

    assert isinstance(mask_collection, list)
    assert len(mask_collection) == num_masks
    for mask in mask_collection:
        assert mask.shape[0] == batch_size
        assert mask.numel() > 0


def test_default_collator_returns_batch_without_masks():
    batch, masks_enc, masks_pred = DefaultCollator()(
        [torch.tensor([1]), torch.tensor([2])]
    )

    assert batch.shape == (2, 1)
    assert masks_enc is None
    assert masks_pred is None


def test_random_mask_collator_splits_patch_indices():
    collator = RandomMaskCollator(ratio=(0.5, 0.5), input_size=(8, 8), patch_size=4)

    batch, masks_enc, masks_pred = collator([torch.zeros(1), torch.ones(1)])

    assert batch.shape == (2, 1)
    assert masks_enc.shape == (2, 2)
    assert masks_pred.shape == (2, 2)
    combined = torch.cat([masks_enc[0], masks_pred[0]]).sort().values.tolist()
    assert combined == [0, 1, 2, 3]


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
    _assert_mask_collection(masks_enc, batch_size=2, num_masks=1)
    _assert_mask_collection(masks_pred, batch_size=2, num_masks=1)
