import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.optim import SGD, AdamW
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.configs import create_parser, get_logger
from datasets.embedding import EmbeddingDataItem
from datasets.embeddingwrapper import EmbeddingDatasetWrapper
from models.adapter import Adapter
from utils.center_loss import Center_loss
from utils.early_stopping import EarlyStopping
from utils.plots import plot_charts
from collections import defaultdict


eer_lr_1em4 = {
    # "teacher_lmaubo": {
    #     # "mipgan1": 0.0,
    #     # "cvmi": 36.35,
    #     "lma": 0.0,
    #     "stylegan": 0.03,
    #     "lmaubo": 0.0,
    #     "mipgan2": 0.03,
    #     "mordiff": 34.21,
    #     "pipe": 34.62,
    # },
    # "teacher_mipgan2": {
    #     # "mipgan1": 0.06,
    #     # "cvmi": 24.52,
    #     "lma": 0.17,
    #     "stylegan": 0.08,
    #     "lmaubo": 0.09,
    #     "mipgan2": 0.11,
    #     "mordiff": 14.6,
    #     "pipe": 14.88,
    # },
    "teacher_mordiff": {
        # "mipgan1": 2.01,
        # "cvmi": 14.04,
        "lma": 0.67,
        "stylegan": 0.87,
        "lmaubo": 3.46,
        "mipgan2": 1.74,
        "mordiff": 0.22,
        "pipe": 0.21,
    },
    "teacher_pipe": {
        # "mipgan1": 4.0,
        # "cvmi": 11.1,
        "lma": 1.45,
        "stylegan": 1.74,
        "lmaubo": 9.04,
        "mipgan2": 2.67,
        "mordiff": 0.29,
        "pipe": 0.36,
    },
    "teacher_lma": {
        "greedy": 14.06,
        "lmaubo": 34.47,
        "mipgan2": 19.72,
        "mordiff": 22.01,
    },
    # {
    #     "stylegan": 0.0,
    #     "Morphing_Diffusion_2024": 14.98,
    #     "mipgan2": 0.04,
    #     "mordiff": 20.24,
    #     "pipe": 20.68,
    # },
    "teacher_lmaubo": {
        "lma": 0.0,
        "stylegan": 0.0,
        "Morphing_Diffusion_2024": 27.7,
        "mipgan2": 0.0,
        "mordiff": 40.11,
        "pipe": 40.39,
    },
    "teacher_post_process": {
        "greedy": 6.56,
        "lmaubo": 12.21,
        "mipgan2": 43.4,
        "mordiff": 7.72,
    },
    # {
    #     "stylegan": 39.73,
    #     "Morphing_Diffusion_2024": 45.46,
    #     "mipgan2": 27.26,
    #     "mordiff": 45.71,
    #     "pipe": 45.69,
    # },
    "teacher_stylegan": {
        "lma": 0.0,
        "lmaubo": 0.0,
        "mordiff": 41.5,
        "pipe": 41.59,
        "stylegan": 0.0,
        "mipgan2": 0.0,
        "Morphing_Diffusion_2024": 34.32,
    },
    "teacher_mipgan2": {
        "lma": 0.0,
        "lmaubo": 0.0,
        "mordiff": 15.77,
        "pipe": 15.96,
        "stylegan": 0.0,
        "mipgan2": 0.0,
        "Morphing_Diffusion_2024": 1.57,
    },
    "teacher_Morphing_Diffusion_2024": {
        "lma": 0.24,
        "lmaubo": 8.41,
        "mordiff": 0.16,
        "pipe": 0.16,
        "stylegan": 0.65,
        "mipgan2": 0.25,
        "Morphing_Diffusion_2024": 0.07,
    },
}

# EER Table: vit with img compression
# 2024-10-23 07:53:17,477 - Eval_26_evaluation - INFO - teacher_lma: {'greedy': 14.06, 'lmaubo': 34.47, 'mipgan2': 19.72, 'mordiff': 22.01}
# teacher_lma: {'greedy': 14.06, 'lmaubo': 34.47, 'mipgan2': 19.72, 'mordiff': 22.01}
# 2024-10-23 07:53:17,477 - Eval_26_evaluation - INFO - teacher_post_process: {'greedy': 6.56, 'lmaubo': 12.21, 'mipgan2': 43.4, 'mordiff': 7.72}
# teacher_post_process: {'greedy': 6.56, 'lmaubo': 12.21, 'mipgan2': 43.4, 'mordiff': 7.72}
# 2024-10-23 07:53:17,477 - Eval_26_evaluation - INFO - teacher_post_process.lmaubo.lma: {'greedy': 10.84, 'lmaubo': 20.28, 'mipgan2': 30.02, 'mordiff': 12.62}
# teacher_post_process.lmaubo.lma: {'greedy': 10.84, 'lmaubo': 20.28, 'mipgan2': 30.02, 'mordiff': 12.62}
# 2024-10-23 07:53:17,477 - Eval_26_evaluation - INFO - student2_post_process_llmbd_1.0: {'greedy': 6.3, 'lmaubo': 20.73, 'mipgan2': 29.83, 'mordiff': 10.51}
# student2_post_process_llmbd_1.0: {'greedy': 6.3, 'lmaubo': 20.73, 'mipgan2': 29.83, 'mordiff': 10.51}

# total_eer_models = {
#     "teacher_lmaubo": 105.24000000000001,
#     "teacher_mipgan2": 54.51,
#     "teacher_mordiff": 23.22,
#     "teacher_pipe": 30.65,
# }
# total = 213.62
# weights = {
#     "teacher_lmaubo": 0.49265050088942985,
#     "teacher_mipgan2": 0.2551727366351465,
#     "teacher_mordiff": 0.10869768748244546,
#     "teacher_pipe": 0.14347907499297818,
# }


def get_weights(teachers: str) -> Dict:
    models = teachers.split(".")
    print(models)
    total_eer_models = defaultdict(float)

    weights = defaultdict(float)

    for model, db_eer_dict in eer_lr_1em4.items():
        for db, eer in db_eer_dict.items():
            total_eer_models[model] += eer

    total = 0
    for model, totals in total_eer_models.items():
        if model[8:] in models:
            total += totals
    print("total: ", total)

    for model, totals in total_eer_models.items():
        if model[8:] in models:
            weights[model] = totals / total

    print("total_eer_models: ", total_eer_models)
    print("weights: ", weights)
    return weights


def get_teacher_embdloaders(teacher_dirs, args) -> List[DataLoader]:
    embdloaders = []
    for dir in teacher_dirs:
        wrapper = EmbeddingDatasetWrapper(root_dir=dir, morph_type=args.morphtype)
        trainembds = wrapper.get_train_embeddings(
            2,
            batch_size=args.batch_size,
            morph_type=args.morphtype,
            student_morph=args.morph_type,
            shuffle=True,
            num_workers=8,
        )
        # testembds = wrapper.get_test_embeddings(
        #     2,
        #     batch_size=args.batch_size,
        #     morph_type=args.morphtype,
        #     shuffle=True,
        #     num_workers=8,
        # )
        # TODO: ??
        embdloaders.append(trainembds)
    return embdloaders


def train(args):
    logger = get_logger("adapter", args.eval_number)
    logger.info("train_adapter")
    early_stopping = EarlyStopping(logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    morphs = args.teacher_morphs.split(".")
    morphs = ["teacher_" + item for item in morphs]

    root_dir = "data/print_scan_digital/embeddings"
    wrapper = EmbeddingDatasetWrapper(root_dir=root_dir, morph_type=args.morphtype)
    train_dataloaders = wrapper.get_multi_dataloaders(
        split_type="train",
        augment_times=2,
        batch_size=args.batch_size,
        morph_types=morphs,
        student_morph=args.morphtype,  # check:
        shuffle=True,
        num_workers=8,
    )

    # change root dir here
    root_dir = "data/print_scan_digital/embeddings"
    teacher_dirs = []
    embd_weight = []
    weights = get_weights(args.teacher_morphs)
    for morph in morphs:
        teacher_dirs.append(f"{root_dir}/teacher_{morph}")
        embd_weight.append(weights[f"teacher_{morph}"])

    # embdloaders = get_teacher_embdloaders(teacher_dirs, args)

    model = Adapter()
    model.initialize_weights()
    model.to(device)
    loss_fn = Center_loss()
    optim = SGD([p for p in model.parameters() if p.requires_grad], args.learning_rate)

    # optim = AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     args.learning_rate,
    #     weight_decay=0.05,
    # )

    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-5)

    dir_path = f"logs/Eval_{args.eval_number}/adapter/checkpoints"

    os.makedirs(dir_path, exist_ok=True)

    chk_pt_path = f"{dir_path}/{args.model_name}_{args.learning_rate}_{args.eval_number}_t1_{morphs[0]}.pt"

    training_loss_list = []
    training_accuracy = []
    epoch = -1
    plot_epoch = 0
    lrs = []
    # max_norm = 0.25

    for epoch in range(epoch + 1, args.epochs):
        dataloaders_iter = [
            iter(dataloader) for dataloader in train_dataloaders.values()
        ]
        model.train()
        train_loss: float = 0.0
        total_correct = 0
        total_samples = 0

        logger.debug(f"Epoch: {epoch}, learning rate:  {args.learning_rate}")
        for batch_idx, batches in enumerate(
            tqdm(
                zip(*dataloaders_iter),
                desc="training",
                position=args.process_num,
            )
        ):
            embeddings = []

            for i, (x, y) in enumerate(batches):
                x = x.to(device) * embd_weight[i]
                embeddings.append(x.unsqueeze(1))

            concatenated_embeddings = torch.cat(embeddings, dim=1)

            representative_labels = batches[0][1].to(device)

            preds = model(concatenated_embeddings)
            batchloss = loss_fn(preds, representative_labels)
            # print("Model output:", preds)
            # print("Loss value:", batchloss.item())
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Layer {name}: Gradient {param.grad.mean().item()}")
            #     else:
            #         print(f"No gradient for {name}")

            optim.zero_grad()

            batchloss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # for name, param in model.named_parameters():
                # if param.grad is not None:
                #     print(f"Gradient for {name}: {param.grad.mean().item()}")
                # else:
                #     print(f"No gradient for {name}")

            optim.step()
            train_loss += batchloss.detach().cpu().item()

            # print(preds.shape, concatenated_labels.shape, concatenated_embeddings.shape)
            # total_correct += (preds == concatenated_embeddings).sum().item()
            # total_samples += concatenated_embeddings.size(0)

        # accuracy = 100 * total_correct / total_samples
        # logger.debug(f"Epoch {epoch+1}: Accuracy = {accuracy:.2f}%")
        logger.debug(f"Train Loss: {train_loss}")
        training_loss_list.append(train_loss)
        # training_accuracy.append(accuracy)

        lrs.append(optim.param_groups[0]["lr"])
        print(f"lrs {lrs}")
        scheduler.step()

        try:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                },
                chk_pt_path,
            )
            logger.debug(f"Model saved successfully at epoch {epoch}")
        except Exception as e:
            logger.debug(f"Error saving model: {e}")
        early_stopping(train_loss)
        plot_epoch = epoch
        if early_stopping.early_stop:
            break

    # logger.debug("train_loss: %s", training_loss_list)
    logger.info("train loss: {}".format(" ".join(map(str, training_loss_list))))
    logger.info("train accuracy: {}".format(" ".join(map(str, training_accuracy))))

    dir_path_charts = f"logs/Eval_{args.eval_number}/adapter/charts"
    plot_charts(
        training_accuracy, [], [], training_loss_list, plot_epoch, args, dir_path_charts
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    train(args)
