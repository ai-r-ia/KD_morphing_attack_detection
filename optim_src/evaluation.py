import os
from tqdm import tqdm
import torch
import pickle
from itertools import combinations

from models.vit import ViTEmbeddings
from utils.eer import compute_eer
from utils.plots import plot_combined_hist, plot_eer_bars, plot_hists
from configs.configs import get_logger, create_parser

# bonafide = 0, morph = 1


def compute_deer(eval_morphs, saving_dir, istest,isferet, logger) -> None:
    # Differential equal eror rate-DEER between dataset pairs

    deer_records = []
    if istest:
        if isferet:
            eer_file = f"{saving_dir}/eer_testdb_feret.pkl"
        else:
            eer_file = f"{saving_dir}/eer_testdb.pkl"
    else:
        eer_file = f"{saving_dir}/eer_traindb.pkl"

    with open(eer_file, "rb") as handle:
        eer_table = pickle.load(handle)

    for model_name, eers in eer_table.items():
        for d1, d2 in combinations(eval_morphs, 2):
            eer1 = eer_table[model_name][d1]
            eer2 = eer_table[model_name][d2]
            deer = abs(eer1 - eer2)
            deer_records.append(
                {"Model": model_name, "Dataset_Pair": f"{d1} vs {d2}", "DEER": deer}
            )

    logger.info("DEER Table:")
    for record in deer_records:
        logger.info(
            f"{record['Model']} on {record['Dataset_Pair']}: DEER = {record['DEER']:.4f}"
        )

    if istest:
        if isferet:
            with open(f"{saving_dir}/deer_testdb_feret.pkl", "wb") as file:
                pickle.dump(deer_records, file)
        else:
            with open(f"{saving_dir}/deer_testdb.pkl", "wb") as file:
                pickle.dump(deer_records, file)
    else:
        with open(f"{saving_dir}/deer_traindb.pkl", "wb") as file:
            pickle.dump(deer_records, file)

    return


def eval(args):
    types = args.teacher_morphs.split(".")
    logger = get_logger("evaluation", args.eval_number)
    # logger.info(f"eval student: {args.student_morph}")

    teacher1 = ViTEmbeddings()
    teacher1.load_state_dict(
        torch.load(
            f"logs/teachers/checkpoints/teacher_0.0001_{types[0]}.pt",
            weights_only=True,
        )["model_state_dict"]
    )

    teacher2 = ViTEmbeddings()
    teacher2.load_state_dict(
        torch.load(
            f"logs/teachers/checkpoints/teacher_0.0001_{types[1]}.pt",
            weights_only=True,
        )["model_state_dict"]
    )

    # teacher3 = ViTEmbeddings()
    # teacher3.load_state_dict(
    #     torch.load(
    #         f"logs/teachers/checkpoints/teacher_0.0001_{types[2]}.pt",
    #         weights_only=True,
    #     )["model_state_dict"]
    # )

    base_morphs = args.teacher_morphs.replace(".", "_")
    if args.student_morph in ["lmaubo", "lma", "post_process"]:
        base_morphs = "lmaubo_lma_post_process"
    if args.student_morph in ["mipgan2", "stylegan", "Morphing_Diffusion_2024"]:
        base_morphs = "mipgan2_stylegan_Morphing_Diffusion_2024"
    baseline = ViTEmbeddings()
    baseline.load_state_dict(
        torch.load(
            f"logs/baseline/checkpoints/baseline_0.0001_{base_morphs}.pt",
            weights_only=True,
        )["model_state_dict"]
    )

    models = {
        f"teacher_{types[0]}": teacher1,
        f"teacher_{types[1]}": teacher2,
        # f"teacher_{types[2]}": teacher3,
        f"teacher_{base_morphs}": baseline,
        # "Adapter": adapter,
        # f"student_{args.student_morph}": student,
    }  # NOTE: special loop for adapter

    loss_lambdas = [1.0]
    students = []
    for i, loss_lambda in enumerate(loss_lambdas):
        students.append(ViTEmbeddings())
        students[i].load_state_dict(
            torch.load(
                f"logs/Eval_{args.eval_number}/student/checkpoints/student_0.01_{args.student_morph}_lmbd_{loss_lambda}_t1_{types[0]}.pt",
                weights_only=True,
            )["model_state_dict"]
        )
        models[f"student_{args.student_morph}_llmbd_{loss_lambda}"] = students[i]

    logger.info("models: {}".format(" ".join(map(str, models.keys()))))

    # eval_morphs = ["mipgan1", "cvmi", "lma", "stylegan"]
    # saved_morphs = ["lmaubo", "mipgan2", "mordiff", "pipe"]
    eval_morphs = args.eval_morphs.split(".")
    saved_morphs = args.teacher_morphs.split(".")
    if not args.istest:
        eval_morphs = saved_morphs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saving_dir = f"logs/Eval_{args.eval_number}"
    os.makedirs(saving_dir, exist_ok=True)
    print(saving_dir)
    compute_eer(
        eval_morphs=eval_morphs,
        models=models,
        device=device,
        saving_dir=saving_dir,
        args=args,
        logger=logger,
    )
    compute_deer(
        eval_morphs=eval_morphs,
        saving_dir=saving_dir,
        istest=args.istest,
        isferet=args.isferet,
        logger=logger,
    )
    # compute_apcer_bpcer(
    #     eval_morphs=eval_morphs, models=models, device=device, saving_dir=saving_dir, args = args
    # )
    # plot_hists(istest=args.istest, saving_dir=saving_dir, models=models.keys())

    # plot_combined_hist(istest=args.istest, saving_dir=saving_dir, models=models.keys())
    plot_eer_bars(istest=args.istest, isferet= args.isferet, saving_dir=saving_dir)

    print("Le Fin")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    eval(args)
