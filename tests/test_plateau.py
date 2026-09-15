from neuralpilot.plateau import detect_plateau


def test_no_plateau_with_insufficient_history():
    assert not detect_plateau([0.5], [])
    assert not detect_plateau([0.5, 0.6], ["model_family"])


def test_plateau_on_small_score_improvement():
    # deltas: 0.50->0.505 (1%), 0.505->0.510 (~0.99%) both < 2%
    assert detect_plateau([0.40, 0.50, 0.505, 0.510], [])


def test_no_plateau_on_large_improvement():
    assert not detect_plateau([0.40, 0.50, 0.65, 0.80], [])


def test_plateau_on_repeated_weakest_block_with_no_gain():
    assert detect_plateau([0.60, 0.60, 0.58], ["hyperparameters", "hyperparameters"])


def test_no_plateau_repeated_block_but_score_gaining():
    assert not detect_plateau([0.50, 0.60, 0.75], ["hyperparameters", "hyperparameters"])


def test_no_plateau_different_weakest_blocks():
    assert not detect_plateau([0.40, 0.55, 0.75], ["hyperparameters", "model_family"])
