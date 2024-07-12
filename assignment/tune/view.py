import sys

import optuna

from assignment.models import check_model


def _load_study(study_name, study_journal_path):
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(study_journal_path),
            ),
        )
        return study
    except KeyError:
        print(
            "error: failed to retrieve tuned hyperparameters (did you perform tuning?)",
            file=sys.stderr,
        )
        return None


def print_nb_trial(trial):
    print(f"Trial number: {trial.number}")
    print(f"Mean validation accuracy: {trial.value}")
    print(f"Parameters: {trial.params}")


def print_nn_trial(trial):
    print(f"Trial number: {trial.number}")
    print(f"Mean validation loss: {trial.value}")
    print(f"Parameters: {trial.params}")


def view_best(model, study_name, study_journal_path) -> bool:
    check_model(model)

    study = _load_study(study_name, study_journal_path)
    if study is None:
        return False

    try:
        if model == "nb":
            print_nb_trial(study.best_trial)
        else:
            assert model == "nn"
            print_nn_trial(study.best_trial)
    except ValueError as ex:
        print(
            f"error: {str(ex)}",
            file=sys.stderr,
        )

    return True


def view_all(model, study_name, study_journal_path) -> bool:
    check_model(model)

    study = _load_study(study_name, study_journal_path)
    if study is None:
        return False

    try:
        trials = study.get_trials()
    except ValueError as ex:
        print(
            f"error: {str(ex)}",
            file=sys.stderr,
        )

    print_trial_f = print_nb_trial if model == "nb" else print_nn_trial
    for trial in trials:
        print_trial_f(trial)
        print()

    return True
