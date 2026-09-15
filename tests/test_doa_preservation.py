import numpy as np

from src.training.eval_doa_preservation import (
    circular_error_deg,
    estimate_doa_srp_phat,
    geometry_aware_error_deg,
    horizontal_geometry_rank,
    reflect_angle_across_rank_one_array,
)


def test_circular_error_wraps_at_360_degrees():
    assert circular_error_deg(359.0, 1.0) == 2.0
    assert circular_error_deg(1.0, 359.0) == 2.0


def test_rank_one_x_axis_resolves_front_back_ambiguity():
    geometry = np.asarray([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]])
    assert horizontal_geometry_rank(geometry) == 1
    assert np.isclose(
        reflect_angle_across_rank_one_array(30.0, geometry), 150.0)
    error, mode = geometry_aware_error_deg(150.0, 30.0, geometry)
    assert mode == 'rank1_resolved'
    assert np.isclose(error, 0.0)


def test_rank_one_y_axis_uses_its_actual_orientation():
    geometry = np.asarray([[0.0, -0.04, 0.0], [0.0, 0.04, 0.0]])
    assert horizontal_geometry_rank(geometry) == 1
    assert np.isclose(
        reflect_angle_across_rank_one_array(30.0, geometry), 330.0)
    error, mode = geometry_aware_error_deg(330.0, 30.0, geometry)
    assert mode == 'rank1_resolved'
    assert np.isclose(error, 0.0)


def test_rank_two_geometry_keeps_full_circular_error():
    geometry = np.asarray([
        [-0.04, -0.04, 0.0],
        [-0.04, 0.04, 0.0],
        [0.04, -0.04, 0.0],
        [0.04, 0.04, 0.0],
    ])
    assert horizontal_geometry_rank(geometry) == 2
    error, mode = geometry_aware_error_deg(150.0, 30.0, geometry)
    assert mode == 'circular'
    assert np.isclose(error, 120.0)


def test_srp_phat_recovers_rank_one_bearing_with_mirror_resolution():
    sample_rate = 8000
    speed = 343.0
    delay_samples = 2
    spacing = speed * delay_samples / sample_rate
    geometry = np.asarray([
        [-spacing / 2.0, 0.0, 0.0],
        [spacing / 2.0, 0.0, 0.0],
    ])
    rng = np.random.default_rng(230)
    source = rng.standard_normal(sample_rate * 2)
    # A source at +x reaches the right microphone first.  Advancing that
    # channel by two samples produces the modeled positive pair delay.
    audio = np.stack([source, np.roll(source, -delay_samples)])
    estimate = estimate_doa_srp_phat(
        audio, geometry, sample_rate=sample_rate,
        speed_of_sound=speed, frame_length=512, hop_length=128,
        frequency_min=200.0, frequency_max=3500.0)
    assert estimate['valid_estimation']
    error, mode = geometry_aware_error_deg(
        estimate['est_azim_deg'], 90.0, geometry)
    assert mode == 'rank1_resolved'
    assert error < 1.0
    assert estimate['n_pairs'] == 1
    assert estimate['n_active_frames'] > 0
