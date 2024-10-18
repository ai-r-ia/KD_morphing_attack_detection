from typing import List
from torch.utils.data import DataLoader
import pickle
from typing import Tuple
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from configs.configs import get_logger, create_parser
from models.adapter import Adapter
from train_adapter import get_weights
from utils.early_stopping import EarlyStopping
from utils.plots import plot_charts
from models.vit import ViTEmbeddingsAdaptive
from datasets.datawrapper import DatasetWrapper
from train_classifier import train_classifier
from datasets.embeddingwrapper import EmbeddingDatasetWrapper


def get_teacher_embdloaders(
    teacher_dirs, args
) -> Tuple[List[DataLoader], List[DataLoader]]:
    train_embdloaders = []
    test_embdloaders = []
    for dir in teacher_dirs:
        wrapper = EmbeddingDatasetWrapper(root_dir=dir, morph_type=args.morphtype)
        trainembds = wrapper.get_train_embeddings(
            2,
            batch_size=args.batch_size,
            morph_type=args.morphtype,
            shuffle=True,
            num_workers=8,
        )
        testembds = wrapper.get_test_embeddings(
            2,
            batch_size=args.batch_size,
            morph_type=args.morphtype,
            shuffle=True,
            num_workers=8,
        )
        train_embdloaders.append(trainembds)
        test_embdloaders.append(testembds)
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
    root_dir = "data/print_scan_digital/embeddings"
    teacher_dirs = []
    embd_weight = []
    for morph in morphs:
        teacher_dirs.append(f"{root_dir}/teacher_{morph}")
        embd_weight.append(weights[f"teacher_{morph}"])

    train_embdloaders, test_embdloaders = get_teacher_embdloaders(teacher_dirs, args)

    loss_fn1 = CrossEntropyLoss()
    loss_fn2 = MSELoss()
    model = ViTEmbeddingsAdaptive()
    optim = SGD([p for p in model.parameters() if p.requires_grad], args.learning_rate)

    dir_path = f"logs/Eval_{args.eval_number}/student2/checkpoints"

    os.makedirs(dir_path, exist_ok=True)

    chk_pt_path = f"{dir_path}/{args.model_name}_{args.learning_rate}_{args.morphtype}_lmbd_{args.lambda_loss}_t1_{morphs[0]}.pt"

    adapter = Adapter()
    chk_pt_path_adapter = f"logs/Eval_{args.eval_number}/adapter/checkpoints/adapter_0.1_{args.eval_number}_t1_{morphs[0]}.pt"
    checkpoint = torch.load(chk_pt_path_adapter, weights_only=True)
    adapter.load_state_dict(checkpoint["model_state_dict"], strict=False)
    adapter.to(device)
    adapter.eval()

    validation_after = 1
    training_loss_list = []
    testing_loss_list = []
    accuracy_list = []
    epoch = -1
    plot_epoch = 0

    for epoch in range(epoch + 1, args.epochs):
        model.to(device)
        # logger.debug(model)
        model.train()
        plot_epoch = epoch
        bon_correct = 0
        bon_incorrect = 0
        mor_correct = 0
        mor_incorrect = 0
        train_loss: float = 0.0

        logger.debug(f"Epoch: {epoch}, learning rate: {args.learning_rate}")

        for batch_idx, (batch1, batch2, batch4) in enumerate(
            tqdm(
                zip(
                    train_embdloaders[0],
                    train_embdloaders[1],
                    # train_embdloaders[2],
                    trainds,
                ),
                desc="training",
                position=args.process_num,
            )
        ):
            x1, y1 = batch1
            x2, y2 = batch2
            # x3, y3 = batch3
            img_path, x4, y4 = batch4
            x1, y1 = x1.to(device), y1.to(device)
            x2, y2 = x2.to(device), y2.to(device)
            # x3, y3 = x3.to(device), y3.to(device)
            x4, y4 = x4.to(device), y4.to(device)

            x1 = x1 * embd_weight[0]
            x2 = x2 * embd_weight[1]
            if x1.shape != x2.shape:
                print("unequal batches")
                continue
            # x3 = x3 * embd_weight[2]
            concatenated_embeddings = torch.cat(
                [
                    x1.unsqueeze(1),
                    x2.unsqueeze(1),
                    #  , x3.unsqueeze(1)
                ],
                dim=1,
            )

            # concatenated_labels = torch.cat([y1, y2, y3], dim=1)

            optim.zero_grad()
            embeddings = adapter(concatenated_embeddings.cuda())
            embeddings = torch.squeeze(embeddings)
            preds, adapter_preds, embds = model(x4, embeddings)
            preds = preds.to(device)

            if embds.shape != embeddings.shape:
                print("unequal embeddings")
                continue

            loss_lambda = adapter_preds[:, 0].mean().detach().cpu()
            
            loss1 = loss_fn1(preds, y4)
            loss2 = loss_fn2(embds, embeddings)
            total_loss = ((1 - loss_lambda) * loss1) + (loss_lambda * loss2)
            total_loss.backward()
            optim.step()

            bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(preds, y4)
            bon_correct += bcorrect
            bon_incorrect += bincorrect
            mor_correct += mcorrect
            mor_incorrect += mincorrect
            train_loss += total_loss.detach().cpu().item()
        logger.debug(f"Train Loss: {train_loss}")
        training_loss_list.append(train_loss)
        logger.debug(
            f"Bonafide ({bon_correct + bon_incorrect}): correct: {bon_correct} incorrect: {bon_incorrect}"
        )
        logger.debug(
            f"Morph ({mor_correct + mor_incorrect}): correct: {mor_correct} incorrect: {mor_incorrect}"
        )

        if not epoch % validation_after:
            validation_loss: float = 0.0
            model.eval()

            # with torch.no_grad():
            for batch_idx, (batch1, batch2, batch4) in enumerate(
                tqdm(
                    zip(
                        test_embdloaders[0],
                        test_embdloaders[1],
                        # test_embdloaders[2],
                        testds,
                    ),
                    desc="testing",
                    position=args.process_num,
                )
            ):
                x1, y1 = batch1
                x2, y2 = batch2
                # x3, y3 = batch3
                img_path, x4, y4 = batch4
                x1, y1 = x1.to(device), y1.to(device)
                x2, y2 = x2.to(device), y2.to(device)
                # x3, y3 = x3.to(device), y3.to(device)
                x4, y4 = x4.to(device), y4.to(device)

                x1 = x1 * embd_weight[0]
                x2 = x2 * embd_weight[1]
                if x1.shape != x2.shape:
                    print("unequal batches")
                    continue
                # x3 = x3 * embd_weight[2]
                concatenated_embeddings = torch.cat(
                    [
                        x1.unsqueeze(1),
                        x2.unsqueeze(1),
                        #   x3.unsqueeze(1)
                    ],
                    dim=1,
                )

                embeddings = adapter(concatenated_embeddings.cuda())
                embeddings = torch.squeeze(embeddings)
                preds, adapter_preds, embds = model(x4, embeddings)
                preds = preds.to(device)

                loss_lambda = adapter_preds[:, 0].mean().detach().cpu()
                loss1 = loss_fn1(preds, y4)
                loss2 = loss_fn2(embds, embeddings)
                total_loss = ((1 - loss_lambda) * loss1) + (loss_lambda * loss2)

                total_loss.backward()
                validation_loss += total_loss.detach().cpu().item()

            logger.debug(f"Test Loss: {validation_loss}")
            testing_loss_list.append(validation_loss)
            logger.debug(
                f"Bonafide ({bon_correct + bon_incorrect}): correct: {bon_correct} incorrect: {bon_incorrect}"  # noqa: E501
            )
            logger.debug(
                f"Morph ({mor_correct + mor_incorrect}): correct: {mor_correct} incorrect: {mor_incorrect}"  # noqa: E501
            )

            accuracy = 100

            if (bon_incorrect + mor_incorrect) > 0:
                accuracy = (
                    (bon_correct + mor_correct) / (bon_incorrect + mor_incorrect)
                ) / 100
                accuracy_list.append(accuracy)

                logger.debug(f"Calculated acc:  {accuracy_list}")

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
                early_stopping(validation_loss)
                if early_stopping.early_stop:
                    break
            else:
                accuracy_list.append(accuracy)
                plot_epoch = epoch
                break

    logger.debug("acc loss: {}".format(" ".join(map(str, accuracy_list))))
    logger.debug("train loss: {}".format(" ".join(map(str, training_loss_list))))
    logger.debug("test loss: {}".format(" ".join(map(str, testing_loss_list))))
    logger.debug(f"PLOT_EPOCH {plot_epoch}")

    dir_path_charts = f"logs/Eval_{args.eval_number}/student2/charts"
    plot_charts(
        accuracy_list,
        testing_loss_list,
        training_loss_list,
        plot_epoch,
        args,
        dir_path_charts,
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    train(args)
