import os
import torch
from torch.optim import SGD, AdamW
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time

from utils.arc_loss import ArcLoss
from configs.configs import get_logger, create_parser
from models.resnet import ResNet34Embeddings
from utils.early_stopping import EarlyStopping
from utils.plots import plot_charts
from models.vit import ViTEmbeddings
from datasets.datawrapper import DatasetWrapper
from train_classifier import train_classifier


def train(args):
    logger = get_logger("teachers_lr_vit_100")
    logger.info(f"train_teacher {args.morphtype}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(logger=logger)

    wrapper = DatasetWrapper(args.root_dir, args.morphtype, morph_dir=args.morph_dir)
    trainds = wrapper.get_train_dataset(
        2, args.batch_size, morph_type=args.morphtype, shuffle=True, num_workers=8
    )
    testds = wrapper.get_test_dataset(
        2, args.batch_size, morph_type=args.morphtype, shuffle=True, num_workers=8
    )

    arc_loss = ArcLoss()
    loss_fn = CrossEntropyLoss()
    model = ViTEmbeddings()
    # if args.model == 'vit':
    #     logger.info("Model is VIT")
    #     model = ViTEmbeddings()
    # elif args.model == 'resnet':
    #     logger.info("Model is Resnet34")
    #     model = ResNet34Embeddings()
    optim = SGD([p for p in model.parameters() if p.requires_grad], args.learning_rate)
    # optim = Adam([p for p in model.parameters() if p.requires_grad], args.learning_rate)
    # optim = AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     args.learning_rate,
    #     weight_decay=0.05,
    # )

    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-5)
    dir_path = "logs/teachers_lr_vit_100/checkpoints"
    os.makedirs(dir_path, exist_ok=True)
    chk_pt_path = (
        f"{dir_path}/{args.model_name}_{args.learning_rate}_{args.morphtype}.pt"
    )

    validation_after = 1
    training_loss_list = []
    testing_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    epoch = -1
    plot_epoch = 0
    lrs = []

    # if os.path.exists(chk_pt_path):
    #     print(f"loading model for {args.morphtype}")
    #     checkpoint = torch.load(chk_pt_path, weights_only=True)
    #     model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    #     epoch = checkpoint["epoch"]
    #     optim.load_state_dict(checkpoint["optimizer_state_dict"])
    #     plot_epoch = epoch + 1

    for epoch in range(epoch + 1, args.epochs):
        start_time = time.time()
        model.to(device)
        model.train()
        plot_epoch = epoch
        bon_correct = 0
        bon_incorrect = 0
        mor_correct = 0
        mor_incorrect = 0
        train_loss: float = 0.0
        total_train_samples = 0
        logger.debug(f"Epoch: {epoch}, learning rate: {args.learning_rate}")

        # Start training loop
        train_start_time = time.time()
        for image_path, x, y in tqdm(trainds, desc="training"):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            preds, embds = model(x)
            bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(preds, y)
            # logits = arc_loss(embds.to(device), y)
            # batchloss = loss_fn(logits, y)
            labels = torch.argmax(y, dim=1)
            batchloss = loss_fn(preds, labels)
            # print("arcloss ", batchloss, "CSE: ", prev_loss)
            batchloss.backward()
            optim.step()

            bon_correct += bcorrect
            bon_incorrect += bincorrect
            mor_correct += mcorrect
            mor_incorrect += mincorrect
            train_loss += batchloss.detach().cpu().item()
            total_train_samples += len(y)

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

        # Testing loop
        test_start_time = time.time()
        if not epoch % validation_after:
            validation_loss: float = 0.0
            total_test_samples = 0
            bon_correct = 0
            bon_incorrect = 0
            mor_correct = 0
            mor_incorrect = 0
            model.eval()

            with torch.no_grad():
                for image_path, x, y in tqdm(testds, desc="testing"):
                    x, y = x.to(device), y.to(device)
                    preds, embds = model(x)
                    bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(
                        preds, y
                    )
                    # logits = arc_loss(embds.to(device), y)
                    # batchloss = loss_fn(logits, y)

                    # batchloss = loss_fn(preds, y)
                    labels = torch.argmax(y, dim=1)
                    batchloss = loss_fn(preds, labels)

                    # print("arcloss ", batchloss, "CSE: ", prev_loss)
                    validation_loss += batchloss.detach().cpu().item()
                    bon_correct += bcorrect
                    bon_incorrect += bincorrect
                    mor_correct += mcorrect
                    mor_incorrect += mincorrect
                    total_test_samples += len(y)

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

            # Save model checkpoint
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

    # Plot charts for accuracy and loss
    dir_path_charts = "logs/teachers_lr_vit_100/charts"
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
