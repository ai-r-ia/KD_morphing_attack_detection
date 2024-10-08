import argparse
import logging
import os


def get_logger(filename: str, eval_num: int = -1) -> logging.Logger:
    # Create a logger for the given file
    logger = logging.getLogger(f"Eval_{eval_num}_{filename}")
    logger.setLevel(logging.DEBUG)
    # if not logger.hasHandlers():

    if eval_num == -1:
        log_dir = f"logs/{filename}"
    else:
        log_dir = f"logs/Eval_{eval_num}"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{log_dir}/{filename}.log")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--morphtype", default="lma", type=str, help="specify morph type"
    )

    parser.add_argument(
        "-lr", "--learning_rate", default=1e-4, type=float, help="specify learning rate"
    )

    parser.add_argument(
        "-mn",
        "--model_name",
        default="teacher",
        type=str,
        help="specify which model to train eg: teacher, student, adapter",
    )

    parser.add_argument(
        "-e", "--epochs", default=100, type=int, help="specify number of epochs"
    )

    parser.add_argument(
        "--process-num", default=0, type=int, help="specify number of epochs"
    )

    parser.add_argument(
        "-lmb",
        "--lambda_loss",
        default=1.0,
        type=float,
        help="specify constant lambda for distance loss  (student)",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="specify batch size",
    )

    parser.add_argument(
        "-ev",
        "--eval_number",
        default=0,
        type=int,
        help="specify eval number",
    )

    parser.add_argument(
        "-t",
        "--teacher_morphs",
        # default="lmaubo_mipgan2_mordiff",
        type=str,
        help="specify teacher morph types",
    )

    parser.add_argument(
        "--istest",
        action="store_true",
        help="set to use test database for evaluation",
    )
    parser.set_defaults(istest=False)

    parser.add_argument(
        "-sm",
        "--student_morph",
        default="pipe",
        type=str,
        help="specify morph type for student",
    )

    parser.add_argument(
        "-rdir",
        "--root_dir",
        default="/home/ubuntu/volume/data/PRINT_SCAN/digital",
        type=str,
        help="specify root directory address",
    )

    parser.add_argument(
        "-mdir",
        "--morph_dir",
        default="/home/ubuntu/volume/data/PRINT_SCAN/digital",
        type=str,
        help="specify morph directory address",
    )

    return parser
