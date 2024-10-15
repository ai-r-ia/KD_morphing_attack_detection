import os
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import time

from configs.configs import get_logger, create_parser
from utils.early_stopping import EarlyStopping
from utils.plots import plot_charts
from models.vit import ViTEmbeddings
from datasets.datawrapper import DatasetWrapper
from train_classifier import train_classifier


def train(args):
    logger = get_logger("baseline")
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

    loss_fn = CrossEntropyLoss()
    model = ViTEmbeddings()
    optim = SGD([p for p in model.parameters() if p.requires_grad], args.learning_rate)

    # Landmark based - Ima, lmaubo, post process
    # GAN - stylegen,  mipgan2,  morphing diffusion 2024, ___cvmi, mipgan1,
    # DIFFUSION - modriff, pipe, greedy,

    teacher_dir_path = (
        # f"logs/baseline/checkpoints/baseline_0.0001_{args.teacher_morphs}.pt"
        f"logs/teachers/checkpoints/teacher_0.0001_{args.teacher_morphs}.pt"
    )
    if os.path.exists(f"{teacher_dir_path}"):
        model.load_state_dict(
            torch.load(
                teacher_dir_path,
                weights_only=True,
            )["model_state_dict"]
        )

    dir_path = "logs/baseline/checkpoints"
    os.makedirs(dir_path, exist_ok=True)

    chk_pt_path = f"{dir_path}/{args.model_name}_{args.learning_rate}_{args.teacher_morphs}_{args.morphtype}.pt"

    validation_after = 1
    training_loss_list = []
    testing_loss_list = []
    accuracy_list = []
    epoch = -1
    plot_epoch = 0

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
        logger.debug(f"Epoch: {epoch},learning rate: {args.learning_rate}")
        train_start_time = time.time()
        for image_path, x, y in tqdm(trainds, desc="training"):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            preds, embds = model(x)
            # logger.debug(preds.shape, y.shape)
            batchloss = loss_fn(preds, y)
            bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(preds, y)
            batchloss.backward()

            optim.step()
            bon_correct += bcorrect
            bon_incorrect += bincorrect
            mor_correct += mcorrect
            mor_incorrect += mincorrect
            train_loss += batchloss.detach().cpu().item()
        logger.debug(f"Train Loss: {train_loss}")
        training_loss_list.append(train_loss)
        logger.debug(
            f"Bonafide ({bon_correct + bon_incorrect}): correct: {bon_correct} incorrect: {bon_incorrect}"
        )
        logger.debug(
            f"Morph ({mor_correct + mor_incorrect}): correct: {mor_correct} incorrect: {mor_incorrect}"
        )
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        logger.info(f"train time : {train_time}")

        test_start_time = time.time()
        if not epoch % validation_after:
            validation_loss: float = 0.0
            model.eval()

            # with torch.no_grad():
            for image_path, x, y in tqdm(testds, desc="testing"):
                x, y = x.to(device), y.to(device)
                preds, embds = model(x)
                batchloss = loss_fn(preds, y)
                batchloss.backward()
                validation_loss += batchloss.detach().cpu().item()
            logger.debug(f"Test Loss: {validation_loss}")
            testing_loss_list.append(validation_loss)
            logger.debug(
                f"Bonafide ({bon_correct + bon_incorrect}): correct: {bon_correct} incorrect: {bon_incorrect}"
            )
            logger.debug(
                f"Morph ({mor_correct + mor_incorrect}): correct: {mor_correct} incorrect: {mor_incorrect}"
            )

            test_end_time = time.time()
            test_time = test_end_time - test_start_time
            logger.info(f"test time : {test_time}")
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

                    end_time = time.time()
                    epoch_time = end_time - start_time
                    logger.debug(f"Epoch time: {epoch_time}")
                except Exception as e:
                    logger.debug(f"Error saving model: {e}")
                early_stopping(validation_loss)
                if early_stopping.early_stop:
                    break
            else:
                accuracy_list.append(accuracy)
                plot_epoch = epoch
                break

    logger.debug(f"acc list: {accuracy_list}")
    logger.debug(f"train_loss:  {training_loss_list}")
    logger.debug(f"test_loss:  {testing_loss_list}")
    logger.debug(f"PLOT_EPOCH {plot_epoch}")

    dir_path_charts = "logs/baseline/charts"
    args.morphtype = f"{args.morphtype}_{args.teacher_morphs}"
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
