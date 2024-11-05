from typing import Any, Dict, List
from torch.utils.data import DataLoader
import pickle
from typing import Tuple
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.optim import SGD, AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from configs.configs import get_logger, create_parser
from models.adapter import Adapter
from models.resnet import ResNet34EmbeddingsAdaptive
from train_adapter import get_weights
from utils.early_stopping import EarlyStopping
from utils.plots import plot_charts
from models.vit import ViTEmbeddingsAdaptive
from datasets.datawrapper import DatasetWrapper
from train_classifier import train_classifier
from datasets.embeddingwrapper import EmbeddingDatasetWrapper


def get_teacher_embdloaders(
    morphs, args
) -> Tuple[Dict[str, DataLoader[Any]], Dict[str, DataLoader[Any]]]:
    train_embdloaders = []
    test_embdloaders = []
    dir = "data/print_scan_digital/embeddings"
    wrapper = EmbeddingDatasetWrapper(root_dir=dir, morph_type=args.morphtype)
    train_embdloaders = wrapper.get_multi_dataloaders(
        split_type="train",
        augment_times=2,
        batch_size=args.batch_size,
        morph_types=morphs,
        student_morph=args.morphtype,  # check:
        shuffle=True,
        num_workers=8,
    )
    test_embdloaders = wrapper.get_multi_dataloaders(
        split_type="test",
        augment_times=2,
        batch_size=args.batch_size,
        morph_types=morphs,
        student_morph=args.morphtype,  # check:
        shuffle=True,
        num_workers=8,
    )
    return train_embdloaders, test_embdloaders


def train(args):
    logger = get_logger("student2", args.eval_number)
    logger.info("train_student2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(logger=logger)

    wrapper = DatasetWrapper(args.root_dir, args.morphtype, morph_dir=args.morph_dir)
    trainds = wrapper.get_train_dataset(
        2, args.batch_size, args.morphtype, shuffle=True, num_workers=8
    )
    testds = wrapper.get_test_dataset(
        2, args.batch_size, args.morphtype, shuffle=True, num_workers=8
    )

    morphs = args.teacher_morphs.split(".")
    logger.info("teachers: {}".format(" ".join(map(str, morphs))))
    logger.info(f"student2 morph type:  {args.morphtype}")

    weights = get_weights(args.teacher_morphs)
    # change root dir here
    embd_weight = []
    for morph in morphs:
        embd_weight.append(weights[f"teacher_{morph}"])

    morphs = ["teacher_" + item for item in morphs]
    train_embdloaders, test_embdloaders = get_teacher_embdloaders(morphs, args)

    loss_fn1 = CrossEntropyLoss()
    loss_fn2 = MSELoss()
    model = ViTEmbeddingsAdaptive()
    # model = ResNet34EmbeddingsAdaptive()
    optim = SGD([p for p in model.parameters() if p.requires_grad], args.learning_rate)
    # optim = AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     args.learning_rate,
    #     weight_decay=0.05,
    # )
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-5)

    dir_path = f"logs/Eval_{args.eval_number}/student2/checkpoints"

    os.makedirs(dir_path, exist_ok=True)

    chk_pt_path = f"{dir_path}/{args.model_name}_{args.learning_rate}_{args.morphtype}_lmbd_{args.lambda_loss}_t1_{morphs[0]}.pt"

    adapter = Adapter()
    adapter.initialize_weights()
    chk_pt_path_adapter = f"logs/Eval_{args.eval_number}/adapter/checkpoints/adapter_0.1_{args.eval_number}_t1_{morphs[0]}.pt"
    checkpoint = torch.load(chk_pt_path_adapter, weights_only=True)
    adapter.load_state_dict(checkpoint["model_state_dict"], strict=False)
    adapter.to(device)
    adapter.eval()

    validation_after = 1
    training_loss_list = []
    testing_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    epoch = -1
    plot_epoch = 0
    lrs = []

    for epoch in range(epoch + 1, args.epochs):
        start_time = time.time()
        model.to(device)
        # logger.debug(model)
        model.train()
        plot_epoch = epoch
        bon_correct = 0
        bon_incorrect = 0
        mor_correct = 0
        mor_incorrect = 0
        train_loss: float = 0.0
        total_train_samples = 0
        logger.debug(f"Epoch: {epoch}, learning rate: {args.learning_rate}")

        train_embdloaders_iter = [
            iter(dataloader) for dataloader in train_embdloaders.values()
        ]
        test_embdloaders_iter = [
            iter(dataloader) for dataloader in test_embdloaders.values()
        ]

        train_start_time = time.time()
        for batch_idx, (batches) in enumerate(
            tqdm(
                zip(
                    *train_embdloaders_iter,
                    iter(trainds),
                ),
                desc="training",
                position=args.process_num,
            )
        ):
            # print(len(batches))
            batch4 = batches[2]
            batches = batches[:2]

            img_path, x4, y4 = batch4
            x4, y4 = x4.to(device), y4.to(device)

            embeddings = []
            for i, (x, y) in enumerate(batches):
                x = x.to(device) * embd_weight[i]
                embeddings.append(x.unsqueeze(1))

            concatenated_embeddings = torch.cat(
                embeddings,
                dim=1,
            )

            optim.zero_grad()
            embeddings = adapter(concatenated_embeddings.cuda())
            embeddings = embeddings.squeeze(1)
            # Pass reshaped embeddings to the model
            preds, adapter_preds, embds = model(x4, embeddings)

            preds = preds.to(device)

            if embds.shape != embeddings.shape:
                print(f"unequal embeddings {embds.shape} and {embeddings.shape}")
                continue

            bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(preds, y4)
            loss_lambda = adapter_preds[:, 0].mean().detach().cpu()
            # print("loss1", preds[:5], y4[:5])
            # print("loss2", embds[:5], embeddings[:5])
            loss1 = loss_fn1(preds, y4)
            loss2 = loss_fn2(embds, embeddings)
            total_loss = ((1 - loss_lambda) * loss1) + (loss_lambda * loss2)
            total_loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Layer {name}: Gradient p{param.grad.mean().item()}")
            #     else:
            #         print(f"No gradient for {name}")
            optim.step()

            bon_correct += bcorrect
            bon_incorrect += bincorrect
            mor_correct += mcorrect
            mor_incorrect += mincorrect
            train_loss += total_loss.detach().cpu().item()
            total_train_samples += len(y4)

        # Calculate train accuracy
        train_accuracy = (bon_correct + mor_correct) / total_train_samples * 100
        train_accuracy_list.append(train_accuracy)
        logger.debug(f"Train Accuracy: {train_accuracy:.2f}%")
        logger.debug(f"Train Loss: {train_loss}")
        training_loss_list.append(train_loss)

        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        logger.info(f"Train time: {train_time:.2f} seconds")

        lrs.append(optim.param_groups[0]["lr"])
        print(f"lrs {lrs}")
        scheduler.step()

        test_start_time = time.time()
        if not epoch % validation_after:
            validation_loss: float = 0.0
            total_test_samples = 0
            bon_correct = 0
            bon_incorrect = 0
            mor_correct = 0
            mor_incorrect = 0
            model.eval()

            # with torch.no_grad():
            for batch_idx, (batches) in enumerate(
                tqdm(
                    zip(
                        *test_embdloaders_iter,
                        iter(testds),
                    ),
                    desc="testing",
                    position=args.process_num,
                )
            ):
                batch4 = batches[2]
                batches = batches[:2]
                embeddings = []
                for i, (x, y) in enumerate(batches):
                    x = x.to(device) * embd_weight[i]
                    embeddings.append(x.unsqueeze(1))
                img_path, x4, y4 = batch4
                x4, y4 = x4.to(device), y4.to(device)

                concatenated_embeddings = torch.cat(
                    embeddings,
                    dim=1,
                )

                embeddings = adapter(concatenated_embeddings.cuda())
                embeddings = embeddings.squeeze(1)
                preds, adapter_preds, embds = model(x4, embeddings)
                preds = preds.to(device)
                if embds.shape != embeddings.shape:
                    print(f"unequal embeddings {embds.shape} and {embeddings.shape}")
                    continue
                bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(preds, y4)
                loss_lambda = adapter_preds[:, 0].mean().detach().cpu()
                loss1 = loss_fn1(preds, y4)
                loss2 = loss_fn2(embds, embeddings)
                total_loss = ((1 - loss_lambda) * loss1) + (loss_lambda * loss2)

                total_loss.backward()
                validation_loss += total_loss.detach().cpu().item()
                bon_correct += bcorrect
                bon_incorrect += bincorrect
                mor_correct += mcorrect
                mor_incorrect += mincorrect
                total_test_samples += len(y4)

            # Calculate test accuracy
            test_accuracy = (bon_correct + mor_correct) / total_test_samples * 100
            test_accuracy_list.append(test_accuracy)
            logger.debug(f"Test Accuracy: {test_accuracy:.2f}%")
            logger.debug(f"Test Loss: {validation_loss}")
            testing_loss_list.append(validation_loss)

            test_end_time = time.time()
            test_time = test_end_time - test_start_time
            logger.info(f"Test time: {test_time:.2f} seconds")

            if test_accuracy >= 100.0:
                logger.info(
                    f"Reached 100% test accuracy at epoch {epoch}. Stopping early."
                )
                plot_epoch = epoch
                break

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                },
                chk_pt_path,
            )
            logger.debug(f"Model saved successfully at epoch {epoch}")

            early_stopping(validation_loss)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered. Ending training.")
                plot_epoch = epoch
                break

        # Epoch timing and logging
        end_time = time.time()
        epoch_time = end_time - start_time
        logger.debug(f"Epoch time: {epoch_time:.2f} seconds")

    logger.debug(f"train acc list: {train_accuracy_list}")
    logger.debug(f"test acc list: {test_accuracy_list}")
    logger.debug(f"train_loss:  {training_loss_list}")
    logger.debug(f"test_loss:  {testing_loss_list}")
    logger.debug(f"PLOT_EPOCH {plot_epoch}")
    logger.debug(f"learning rates: {lrs}")

    dir_path_charts = f"logs/Eval_{args.eval_number}/student2/charts"
    plot_charts(
        train_accuracy_list,
        test_accuracy_list,
        training_loss_list,
        testing_loss_list,
        plot_epoch,
        args,
        dir_path_charts,
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    train(args)
