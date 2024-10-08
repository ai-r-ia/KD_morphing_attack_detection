from typing import List
from multiprocessing import Pool
import os
import numpy as np
import random
import torch
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

print(sys.path)


# list of syscalls


def teacher_syscall(morph):
    os.system(f"python train_teacher.py -lr 1e-4 -m {morph} -mn teacher")


def adapter_syscall(lr, teachers: str, eval_number):
    os.system(
        f"python train_adapter.py -lr {lr} -mn adapter -t {teachers} -ev {eval_number}"
    )


def student_syscall(lr, loss_lambda, student_morph, eval_number):
    os.system(
        f"python train_student.py -lr {lr} -m {student_morph} -mn student -lmb {loss_lambda} -ev {eval_number}"
    )


def eval_syscall(istest: bool, eval_number, student_morph):
    if istest:
        os.system(
            f"python evaluation.py -ev {eval_number} -sm {student_morph} --istest"
        )
    else:
        os.system(f"python evaluation.py -ev {eval_number} -sm {student_morph}")


def embds_syscall(teacher_morph: str, morph: str):
    os.system(
        f"python ./optim_src/utils/save_embeddings.py -t {teacher_morph} -m {morph}"
    )


# /home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/ria/mad_kd/optim_src/configs/config.py
# /home/ubuntu/volume/mad_kd/optim_src/configs/config.py
def run_single_iteration(
    i, morphs, learning_rate_adapter, learning_rate_student, loss_lambdas, gpu_semaphore
):
    print(f"Eval {i}")
    teachers = f"{morphs[i % len(morphs)]}_{morphs[(i + 1) % len(morphs)]}_{morphs[(i + 2) % len(morphs)]}"
    student_morph = f"{morphs[(i + 3) % len(morphs)]}"
    print("Teachers: ", teachers)
    print("Student: ", student_morph)

    # with gpu_semaphore:
    #     eval_syscall(istest=True, eval_number=i, student_morph=student_morph)
    if i == 0:
        with gpu_semaphore:
            eval_syscall(istest=False, eval_number=i, student_morph=student_morph)
    # print(f"adapter training_eval_{i}")

    # with gpu_semaphore:
    #     adapter_syscall(lr=learning_rate_adapter, teachers=teachers, eval_number=i)

    # for loss_lambda in loss_lambdas:
    #     print(f"student {student_morph} lambda {loss_lambda}_eval_{i}")
    #     with gpu_semaphore:
    #         student_syscall(
    #             lr=learning_rate_student,
    #             loss_lambda=loss_lambda,
    #             student_morph=student_morph,
    #             eval_number=i,
    #         )


def pool_init(semaphore):
    global pool_semaphore
    pool_semaphore = semaphore


def main():
    learning_rate_adapter = 1e-1
    learning_rate_student = 1e-2
    loss_lambdas = [1, 0.5, 0]  # change b/w 0, 0.5, 1

    morphs = ["lmaubo", "mipgan2", "mordiff", "pipe"]
    # args: List[str] = []
    # process_num = 1
    # for teacher in ["lmaubo", "mipgan2", "mordiff", "pipe"]:
    #     for morph in morphs:
    #         args.append(
    #             f"python ./utils/save_embeddings.py -t {teacher} -m {morph} --process-num={process_num}"
    #         )
    #         process_num += 2

    # args: List[str] = []
    # process_num = 1
    # for i in range(0, len(morphs)):
    #     teachers = f"{morphs[i % len(morphs)]}_{morphs[(i + 1) % len(morphs)]}_{morphs[(i+2) % len(morphs)]}"
    #     student_morph = f"{morphs[(i+3)%len(morphs)]}"

    #     for morph in morphs:
    #     args.append(
    #         f"python train_adapter.py -lr {learning_rate_adapter} -mn adapter -t {teachers} -ev {i} --process-num={process_num} -m {morph}"
    #     )
    #     process_num += 1

    # args: List[str] = []
    # process_num = 1
    # for i in range(0, len(morphs)):
    #     teachers = f"{morphs[i % len(morphs)]}_{morphs[(i + 1) % len(morphs)]}_{morphs[(i+2) % len(morphs)]}"
    #     student_morph = f"{morphs[(i+3)%len(morphs)]}"
    #     for loss_lambda in loss_lambdas:
    #         args.append(
    #             f"python train_student.py -lr {learning_rate_student} -m {student_morph} -mn student -lmb {loss_lambda} -ev {i} -t {teachers} --process-num={process_num}"
    #         )
    #     process_num += 1

    # args: List[str] = []
    # process_num = 1
    # for i in range(0, len(morphs)):
    #     teachers = f"{morphs[i % len(morphs)]}_{morphs[(i + 1) % len(morphs)]}_{morphs[(i+2) % len(morphs)]}"
    #     student_morph = f"{morphs[(i+3)%len(morphs)]}"
    #     args.append(
    #         f"python evaluation.py  -sm {student_morph} -mn student  -ev {i} -t {teachers} --process-num={process_num} --istest"
    #     )
    #     args.append(
    #         f"python evaluation.py  -sm {student_morph} -mn student  -ev {i} -t {teachers} --process-num={process_num}"
    #     )
    #     process_num += 2

    args: List[str] = []
    process_num = 1
    # for morph in ["lma", "post_process", "stylegan", "Morphing_Diffusion_2024"]:
    for morph in [
        # "lmaubo",
        # "mipgan2",
        # "mordiff",
        "pipe",
        "lma",
        "post_process",
        "stylegan",
        "Morphing_Diffusion_2024",
    ]:
        if morph == "post_process":
            morph_dir = "/home/ubuntu/volume/data/PostProcess_Data/Digital/After/Mor"
            args.append(
                f"python train_teacher.py -lr 1e-5 -m {morph} -mn teacher -mdir {morph_dir} --process-num={process_num}"
            )
        else:
            args.append(
                f"python train_teacher.py -lr 1e-5 -m {morph} -mn teacher --process-num={process_num}"
            )
        process_num += 1

    with Pool(3) as pool:
        pool.map(os.system, args)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    # gc.collect()
    # torch.cuda.empty_cache()
    # set_seed()
    main()
