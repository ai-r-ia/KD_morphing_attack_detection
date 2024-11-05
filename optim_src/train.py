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


# /home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/ria/mad_kd/optim_src/configs/config.py
# /home/ubuntu/volume/mad_kd/optim_src/configs/config.py
def pool_init(semaphore):
    global pool_semaphore
    pool_semaphore = semaphore


def main():
    learning_rate_adapter = 1e-3
    learning_rate_student = 1e-3
    args: List[str] = []
    process_num = 1
    EVAL_NUM = 28
    # eval 8 on wards it's cleaned_data, 4-7 previous
    # eval 14-19 is lr scheduler and combined baseline training
    # eval 20 -25  uses resnet with lr scheduler + img compression
    # eval 26 - vit+img compression 60-100, student lr 1e-4, sgd
    # eval 27 - vit, compression 50-100, student lr 1e-3, seed 42
    # eval 28 - same as above, seed 100

    # TRAIN TEACHERS
    # for morph in [
    #     # "lmaubo",
    #     # "mipgan2",
    #     # "mordiff",
    #     # "pipe",
    #     "lma",
    #     "post_process",
    #     # "stylegan",
    #     # "Morphing_Diffusion_2024",
    # ]:
    #     if morph == "post_process":
    #         morph_dir = "/home/ubuntu/volume/data/PostProcess_Data/digital/morph/after"
    #         # morph_dir = "/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/postprocessdata/digital_estimate/morph/after"
    #         args.append(
    #             f"python train_teacher.py -lr 1e-4 -m {morph} -mn teacher -mdir {morph_dir} --process-num={process_num}"
    #         )
    #     else:
    #         args.append(
    #             f"python train_teacher.py -lr 1e-4 -m {morph} -mn teacher --process-num={process_num}"
    #         )
    #     process_num += 1

    # TRAIN BASELINE // modify morphs and run multiple times based on number of teachers
    # for morph in [
    #     # ["lma", "lmaubo"],
    #     # ["stylegan", "mipgan2"],
    #     "post_process.lmaubo.lma",
    #     # "Morphing_Diffusion_2024.mipgan2.stylegan",
    #     # ["mordiff", "pipe"],
    #     # "post_process",
    #     # "Morphing_Diffusion_2024",
    # ]:
    #     args.append(
    #         f"python train_baseline_combined.py -lr 1e-4 -t {morph} -mn baseline --process-num={process_num}"
    #     )
    #     process_num += 1

    # SAVE TEACHER EMBEDDINGS
    # morphs = [
    #     "lmaubo",
    #     # "mipgan2",
    #     # "mordiff",
    #     # "pipe",
    #     # "lma",
    #     # "post_process",
    #     # "stylegan",
    #     # "Morphing_Diffusion_2024",
    # ]
    # for teacher in [
    #     # "lmaubo",
    #     # "mipgan2",
    #     # "mordiff",
    #     # "pipe",
    #     "lma",
    #     "post_process",
    #     # "stylegan",
    #     # "Morphing_Diffusion_2024",
    # ]:
    #     for morph in morphs:
    #         args.append(
    #             f"python ./utils/save_embeddings.py -t {teacher} -m {morph} --process-num={process_num}"
    #         )
    #         process_num += 2

    # TRAIN ADAPTER
    # morphs = [
    #     ["lma", "post_process", "lmaubo"],
    #     # ["stylegan", "Morphing_Diffusion_2024", "mipgan2"],
    # ]
    eval_num = EVAL_NUM
    # for mtypes in morphs:
    #     for i in range(0, len(mtypes)):
    #         teachers = f"{mtypes[i%len(mtypes)]}.{mtypes[(i+1)%len(mtypes)]}"

    # for morph in mtypes:
    # print(morph)
    #     args.append(
    #         f"python train_adapter.py -lr {learning_rate_adapter} -mn adapter -t {teachers} -ev {eval_num} --process-num={process_num} -m {morph}"
    #     )
    #     process_num += 1
    # eval_num += 1

    # teachers = "lma.post_process"
    # morph = "lmaubo"
    # args.append(
    #     f"python train_adapter.py -lr {learning_rate_adapter} -mn adapter -t {teachers} -ev {eval_num} --process-num={process_num} -m {morph} -bs 16"
    # )

    # # TRAIN STUDENT
    # morphs = [
    #     ["lma", "post_process", "lmaubo"],
    #     ["stylegan", "Morphing_Diffusion_2024", "mipgan2"],
    # ]
    eval_num = EVAL_NUM
    # loss_lambdas = [1.0]
    # for mtypes in morphs:
    #     for i in range(0, len(mtypes)):
    #         teachers = f"{mtypes[i%len(mtypes)]}.{mtypes[(i+1)%len(mtypes)]}"
    #         student_morph = f"{mtypes[(i+2)%len(mtypes)]}"
    #         if student_morph == "post_process":
    #             morph_dir = (
    #                 "/home/ubuntu/volume/data/PostProcess_Data/digital/morph/after"
    #             )
    #             for loss_lambda in loss_lambdas:
    #                 args.append(
    #                     f"python train_student2.py -lr {learning_rate_student} -m {student_morph} -mn student -mdir {morph_dir} -lmb {loss_lambda} -ev {eval_num} -t {teachers} --process-num={process_num}"
    #                 )
    #         else:
    #             for loss_lambda in loss_lambdas:
    #                 args.append(
    #                     f"python train_student2.py -lr {learning_rate_student} -m {student_morph} -mn student  -lmb {loss_lambda} -ev {eval_num} -t {teachers} --process-num={process_num}"
    #                 )
    #         process_num += 1
    #         eval_num += 1

    # student_morph = "lmaubo"
    # teachers = "lma.post_process"
    # morph_dir = "/home/ubuntu/volume/data/PostProcess_Data/digital/morph/after"
    # loss_lambda = 1.0
    # args.append(
    #     f"python train_student2.py -lr {learning_rate_student} -m {student_morph} -mn student -mdir {morph_dir} -lmb {loss_lambda} -ev {eval_num} -t {teachers} --process-num={process_num}"
    # )

    # # EVALUATION
    # morphs = [
    #     [
    #         "lma.post_process.lmaubo",
    #         # "stylegan.Morphing_Diffusion_2024.mipgan2.mordiff.pipe",
    #         "greedy.lmaubo.mipgan2.mordiff",
    #     ],
    # [
    #     "stylegan.Morphing_Diffusion_2024.mipgan2",
    #     # "lma.lmaubo.post_process.mordiff.pipe",
    #     "greedy.lmaubo.mipgan2.mordiff",
    # ],
    # ]
    eval_num = EVAL_NUM
    # for morph in morphs:
    #     mtypes = morph[0].split(".")
    #     for i in range(0, len(mtypes)):
    #         teachers = f"{mtypes[i%len(mtypes)]}.{mtypes[(i+1)%len(mtypes)]}"
    #         student_morph = f"{mtypes[(i+2)%len(mtypes)]}"
    #         eval_morphs = morph[1]
    #         # E:\filestorage\nbl-users\Shreyas-Sushrut-Raghu\FaceMoprhingDatabases\cleaned_datasets\feret\digital
    #         root_dir = "/home/ubuntu/volume/data/feret"
    #         args.append(
    #             f"python evaluation.py -rdir {root_dir} -sm {student_morph} -ev {eval_num} -t {teachers} --process-num={process_num} -em {eval_morphs} -bs 128 --istest --isferet"
    #         )
    #         process_num += 1
    #         # args.append(
    #         #     f"python evaluation.py -sm {student_morph} -ev {eval_num} -t {teachers} --process-num={process_num} -em {teachers} -bs 128"
    #         # )
    #         # process_num += 2
    #         eval_num += 1

    student_morph = "lmaubo"
    teachers = "lma.post_process"
    eval_morphs = "greedy.lmaubo.mipgan2.mordiff"
    root_dir = "/home/ubuntu/volume/data/feret"
    args.append(
        f"python evaluation.py -rdir {root_dir} -sm {student_morph} -ev {eval_num} -t {teachers} --process-num={process_num} -em {eval_morphs} -bs 128 --istest --isferet"
    )
    process_num += 1

    with Pool(10) as pool:
        pool.map(os.system, args)


def set_seed(seed: int = 100) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    # gc.collect()
    # torch.cuda.empty_cache()
    set_seed()
    main()
