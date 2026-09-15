import random
import wave

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.training.dataset import (
    LibriSpeechTSEDataset,
    array_agnostic_collate_fn,
)


TENSOR_KEYS = (
    'mixture', 'target', 'enrollment', 'azim_vec', 'mic_mask', 'mic_xyz')


def _write_wav(path, speaker_idx, utterance_idx, sample_rate=8000):
    sample_idx = np.arange(128, dtype=np.int32)
    samples = (
        (sample_idx * (speaker_idx + 3) + utterance_idx * 19) % 127 - 63
    ) * 400
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype('<i2').tobytes())


@pytest.fixture
def tiny_librispeech(tmp_path):
    subset = 'tiny-clean'
    for speaker_idx, speaker_id in enumerate(('100', '200', '300')):
        chapter_dir = tmp_path / subset / speaker_id / '1'
        chapter_dir.mkdir(parents=True)
        for utterance_idx in range(3):
            _write_wav(
                chapter_dir / ('%s-1-%04d.wav' %
                               (speaker_id, utterance_idx)),
                speaker_idx,
                utterance_idx,
            )
    return tmp_path, subset


def _make_dataset(tiny_librispeech, dset, random_seed):
    input_dir, subset = tiny_librispeech
    return LibriSpeechTSEDataset(
        input_dir=str(input_dir),
        subsets=[subset],
        dset=dset,
        sr=8000,
        win=16,
        n_mics=4,
        min_n_mics=3,
        max_n_mics=4,
        variable_n_mics=True,
        return_array_metadata=True,
        geometry_types=['random'],
        seen_geometry_types=['random'],
        geometry_split='seen',
        random_aperture_range=(0.05, 0.2),
        sensor_position_error_std=0.01,
        n_interferers=1,
        segment_len=0.008,
        enrollment_len=0.006,
        azim_type='onehot',
        num_samples=6,
        use_reverb=False,
        random_seed=random_seed,
    )


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _observable_tensors_differ(left, right):
    return any(not torch.equal(left[key], right[key]) for key in TENSOR_KEYS)


def test_train_same_seed_and_epoch_is_exactly_reproducible(
        tiny_librispeech):
    first_dataset = _make_dataset(tiny_librispeech, 'train', 230)
    second_dataset = _make_dataset(tiny_librispeech, 'train', 230)
    first_dataset.set_epoch(3)
    second_dataset.set_epoch(3)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    first_sample = first_dataset[2]
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    second_sample = second_dataset[2]

    _assert_nested_equal(first_sample, second_sample)
    assert torch.count_nonzero(first_sample['mic_xyz']) > 0


def test_train_changes_with_epoch_and_seed(tiny_librispeech):
    dataset = _make_dataset(tiny_librispeech, 'train', 230)
    dataset.set_epoch(0)
    epoch_zero = [dataset[idx] for idx in range(3)]
    dataset.set_epoch(1)
    epoch_one = [dataset[idx] for idx in range(3)]

    other_seed_dataset = _make_dataset(tiny_librispeech, 'train', 231)
    other_seed_dataset.set_epoch(0)
    other_seed = [other_seed_dataset[idx] for idx in range(3)]

    assert any(_observable_tensors_differ(left, right)
               for left, right in zip(epoch_zero, epoch_one))
    assert any(_observable_tensors_differ(left, right)
               for left, right in zip(epoch_zero, other_seed))


def test_validation_is_fixed_across_epochs_and_global_rng(tiny_librispeech):
    first_dataset = _make_dataset(tiny_librispeech, 'val', 42)
    second_dataset = _make_dataset(tiny_librispeech, 'val', 42)
    first_dataset.set_epoch(0)
    second_dataset.set_epoch(99)

    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    first_samples = [first_dataset[idx] for idx in range(3)]
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    second_samples = [second_dataset[idx] for idx in range(3)]

    for first_sample, second_sample in zip(first_samples, second_samples):
        _assert_nested_equal(first_sample, second_sample)
        assert second_sample['metadata']['dataset_epoch'] == 0


def test_worker_count_does_not_change_ordered_samples(tiny_librispeech):
    single_worker_dataset = _make_dataset(tiny_librispeech, 'train', 230)
    multi_worker_dataset = _make_dataset(tiny_librispeech, 'train', 230)
    single_worker_dataset.set_epoch(4)
    multi_worker_dataset.set_epoch(4)

    def make_loader(dataset, num_workers):
        generator = torch.Generator()
        generator.manual_seed(1234)
        return DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=array_agnostic_collate_fn,
            generator=generator,
        )

    single_worker_batches = list(make_loader(single_worker_dataset, 0))
    multi_worker_batches = list(make_loader(multi_worker_dataset, 2))

    _assert_nested_equal(single_worker_batches, multi_worker_batches)


def test_distributed_sampler_is_seeded_per_experiment_and_epoch():
    dataset = range(17)

    def sampled_indices(seed, epoch):
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=True,
            seed=seed,
        )
        sampler.set_epoch(epoch)
        return list(sampler)

    assert sampled_indices(230, 3) == sampled_indices(230, 3)
    assert sampled_indices(230, 3) != sampled_indices(231, 3)
    assert sampled_indices(230, 3) != sampled_indices(230, 4)
