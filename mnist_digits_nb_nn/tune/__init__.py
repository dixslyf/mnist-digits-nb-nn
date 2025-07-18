import numpy as np
import optuna

import mnist_digits_nb_nn.tune
import mnist_digits_nb_nn.tune.view
from mnist_digits_nb_nn.data_loaders import NBDataLoader
from mnist_digits_nb_nn.models import check_model


def _check_args(model, mode):
    check_model(model)

    if mode not in ("tune", "view-best", "view-all"):
        raise ValueError(f'Invalid mode "{mode}"')


def tune(
    model,
    data_dir,
    journal_path,
    study_name,
    trials,
    folds,
    jobs,
    random_state,
    device,
):
    # Load the train and validation sets.
    X_train, y_train = NBDataLoader(data_dir, "train").load()
    X_val, y_val = NBDataLoader(data_dir, "val").load()
    assert X_train.shape[1] == X_val.shape[1]

    # Since we will be performing K-fold cross-validation,
    # we can combine the train and validation sets.
    # In theory, using K-fold cross-validation should
    # reduce the variance of our performance estimates.
    X = np.concatenate((X_train, X_val), axis=0)
    assert X.shape[0] == (X_train.shape[0] + X_val.shape[0])

    y = np.concatenate((y_train, y_val), axis=0)
    assert y.shape == (X.shape[0],)

    if model == "nb":
        mnist_digits_nb_nn.tune.nb.tune_nb(
            X,
            y,
            n_trials=trials,
            n_folds=folds,
            n_jobs=jobs,
            random_state=random_state,
            journal_path=journal_path,
            study_name=study_name,
        )
    else:
        mnist_digits_nb_nn.tune.nn.tune_nn(
            X,
            y,
            n_trials=trials,
            n_folds=folds,
            n_jobs=jobs,
            device=device,
            random_state=random_state,
            journal_path=journal_path,
            study_name=study_name,
        )

    return 0


def cli_entry(
    model,
    data_dir,
    mode,
    journal_path,
    study_name,
    trials,
    folds,
    jobs,
    random_state,
    device,
) -> int:
    _check_args(model, mode)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if mode == "tune":
        return tune(
            model,
            data_dir,
            journal_path,
            study_name,
            trials,
            folds,
            jobs,
            random_state,
            device,
        )
    elif mode == "view-best":
        return (
            0
            if mnist_digits_nb_nn.tune.view.view_best(model, study_name, journal_path)
            else 1
        )
    else:
        assert mode == "view-all"
        return (
            0
            if mnist_digits_nb_nn.tune.view.view_all(model, study_name, journal_path)
            else 1
        )
