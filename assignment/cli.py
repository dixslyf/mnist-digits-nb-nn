import argparse
import os
from typing import Final

import assignment.cv
import assignment.test
import assignment.train
import assignment.tune
from assignment.tune.nb import (DEFAULT_NB_STUDY_JOURNAL_PATH,
                                DEFAULT_NB_STUDY_NAME)
from assignment.tune.nn import (DEFAULT_NN_STUDY_JOURNAL_PATH,
                                DEFAULT_NN_STUDY_NAME)

DATA_PATH: Final[str] = "data"
DIGIT_DATA_PATH: Final[str] = os.path.join(DATA_PATH, "digitdata")


def init_train_parser(parser, base_parser):
    subparsers = parser.add_subparsers(
        help="the model to train", dest="model", required=True
    )

    nb_parser = subparsers.add_parser(
        "nb",
        parents=[base_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    nb_parser.add_argument(
        "-o",
        "--output-path",
        help="the file path to save the trained model to",
        type=str,
        default="nb.pkl",
    )

    nn_parser = subparsers.add_parser(
        "nn",
        parents=[base_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    nn_parser.add_argument(
        "-b",
        "--batch-size",
        help="the number of samples in each batch",
        type=int,
        default=64,
    )

    nn_parser.add_argument(
        "-e",
        "--epochs",
        help="the number of epochs to train the model for",
        type=int,
        default=10,
    )

    nn_parser.add_argument(
        "-l",
        "--learning-rate",
        help="the learning rate",
        type=float,
        default=1.0,
    )

    nn_parser.add_argument(
        "-o",
        "--output-path",
        help="the file path to save the trained model to",
        type=str,
        default="nn.pt",
    )


def init_test_parser(parser, base_parser):
    subparsers = parser.add_subparsers(
        help="the model to test", dest="model", required=True
    )

    nb_parser = subparsers.add_parser(
        "nb",
        parents=[base_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    nb_parser.add_argument(
        "-i",
        "--input-path",
        help="the file path to load the trained model from",
        type=str,
        default="nb.pkl",
    )

    nn_parser = subparsers.add_parser(
        "nn",
        parents=[base_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    nn_parser.add_argument(
        "-i",
        "--input-path",
        help="the file path to load the trained model from",
        type=str,
        default="nn.pt",
    )


def init_cv_parser(parser, base_parser):
    base_cv_parser = argparse.ArgumentParser(add_help=False)

    base_cv_parser.add_argument(
        "-o",
        "--outer-folds",
        help="the number of outer folds",
        type=int,
        default=5,
    )

    base_cv_parser.add_argument(
        "-i",
        "--inner-folds",
        help="the number of inner folds",
        type=int,
        default=5,
    )

    base_cv_parser.add_argument(
        "-t",
        "--trials",
        help="the number of trials for hyperparameter tuning",
        type=int,
        default=25,
    )

    base_cv_parser.add_argument(
        "-j",
        "--jobs",
        help="the number of parallel jobs",
        type=int,
        default=1,
    )

    subparsers = parser.add_subparsers(
        help="the model to perform cross-validation for", dest="model", required=True
    )

    _ = subparsers.add_parser(
        "nb",
        parents=[base_parser, base_cv_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _ = subparsers.add_parser(
        "nn",
        parents=[base_parser, base_cv_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def init_tune_parser(parser, base_parser):
    base_tune_parser = argparse.ArgumentParser(add_help=False)

    base_tune_parser.add_argument(
        "-m",
        "--mode",
        help='the mode to run in ("tune", "view-best" or "view-all")',
        type=str,
        choices=["tune", "view-best", "view-all"],
        default="view-best",
    )

    base_tune_parser.add_argument(
        "-t",
        "--trials",
        help="the number of trials for hyperparameter tuning",
        type=int,
        default=25,
    )

    base_tune_parser.add_argument(
        "-f",
        "--folds",
        help="the number of folds for K-fold cross-validation",
        type=int,
        default=5,
    )

    base_tune_parser.add_argument(
        "-j",
        "--jobs",
        help="the number of parallel jobs",
        type=int,
        default=1,
    )

    subparsers = parser.add_subparsers(
        help="the model to tune", dest="model", required=True
    )

    nb_parser = subparsers.add_parser(
        "nb",
        parents=[base_parser, base_tune_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    nb_parser.add_argument(
        "-l",
        "--journal-path",
        help="the path to the Optuna journal to load and view tuned parameters",
        type=str,
        default=DEFAULT_NB_STUDY_JOURNAL_PATH,
    )

    nb_parser.add_argument(
        "-n",
        "--study-name",
        help="the name of the Optuna study to load from the journal",
        type=str,
        default=DEFAULT_NB_STUDY_NAME,
    )

    nn_parser = subparsers.add_parser(
        "nn",
        parents=[base_parser, base_tune_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    nn_parser.add_argument(
        "-l",
        "--journal-path",
        help="the path to the Optuna journal to load and view tuned parameters",
        type=str,
        default=DEFAULT_NN_STUDY_JOURNAL_PATH,
    )

    nn_parser.add_argument(
        "-n",
        "--study-name",
        help="the name of the Optuna study to load from the journal",
        type=str,
        default=DEFAULT_NN_STUDY_NAME,
    )


def make_parser() -> argparse.ArgumentParser:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "-d",
        "--data-dir",
        help="the data set folder path",
        type=str,
        default=DIGIT_DATA_PATH,
    )

    base_parser.add_argument(
        "-r",
        "--random-state",
        help="the random state for reproducible results",
        type=int,
        default=0,
    )

    parser = argparse.ArgumentParser(
        prog="assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        help="the subcommand to run", dest="subcommand", required=True
    )

    train_parser = subparsers.add_parser("train")
    init_train_parser(train_parser, base_parser)

    test_parser = subparsers.add_parser("test")
    init_test_parser(test_parser, base_parser)

    cv_parser = subparsers.add_parser("cv")
    init_cv_parser(cv_parser, base_parser)

    tune_parser = subparsers.add_parser("tune")
    init_tune_parser(tune_parser, base_parser)

    return parser


def run():
    parser = make_parser()
    args = parser.parse_args()
    match args.subcommand:
        case "train":
            return assignment.train.cli_entry(
                model=args.model,
                data_dir=args.data_dir,
                output_path=args.output_path,
                random_state=args.random_state,
            )
        case "test":
            return assignment.test.cli_entry(
                model=args.model,
                data_dir=args.data_dir,
                input_path=args.input_path,
                random_state=args.random_state,
            )
        case "cv":
            return assignment.cv.cli_entry(
                model=args.model,
                data_dir=args.data_dir,
                trials=args.trials,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                jobs=args.jobs,
                random_state=args.random_state,
            )
        case "tune":
            return assignment.tune.cli_entry(
                model=args.model,
                data_dir=args.data_dir,
                mode=args.mode,
                journal_path=args.journal_path,
                study_name=args.study_name,
                trials=args.trials,
                folds=args.folds,
                jobs=args.jobs,
                random_state=args.random_state,
            )
