import torch
from tqdm import tqdm
import numpy as np
import os

from datasets.datawrapper import DatasetWrapper
from models.resnet import ResNet34Embeddings
from models.vit import ViTEmbeddings
from configs.configs import create_parser, get_logger


def save_single_teacher_embds(
    morph: str,
    dir_path: str,
    data_loader,
    dataset_type: str,
    data_morphtype: str,
    logger,
    process_num: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model = ViTEmbeddings(only_embeddings=True)
    # teacher_model = ResNet34Embeddings(only_embeddings=True)
    chk_pt_path_tchr = f"logs/teachers_lr_vit_100/checkpoints/teacher_0.0001_{morph}.pt"
    checkpoint = torch.load(chk_pt_path_tchr, weights_only=True)
    teacher_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    teacher_model.to(device)
    teacher_model.eval()

    embeddings = []
    with torch.no_grad():
        for image_name, x, y in tqdm(data_loader, position=process_num):
            x, y = x.to(device), y.cpu().numpy()
            embds = teacher_model(x)
            embeddings = embds.detach().cpu().numpy()
            # print(embeddings.shape)

            for i in range(len(embeddings)):
                label = "bonafide" if y[i].argmax() == 0 else "morph"
                if label == "bonafide":
                    save_dir = os.path.join(
                        dir_path, f"teacher_{morph}", label, dataset_type
                    )
                else:
                    save_dir = os.path.join(
                        dir_path,
                        f"teacher_{morph}",
                        label,
                        data_morphtype,
                        dataset_type,
                    )
                os.makedirs(save_dir, exist_ok=True)

                base_name = os.path.splitext(os.path.basename(image_name[i]))[0]
                embedding_file_path = os.path.join(save_dir, f"{base_name}.npy")
                np.save(embedding_file_path, embeddings[i])

    logger.info(
        f"saving {dataset_type} embeddings for teacher og on {morph}, now on {args.morphtype}"
    )


def save(args) -> None:
    logger = get_logger("save_embeddings")
    morph = args.teacher_morphs
    if args.morphtype == "post_process" or morph == "post_process":
        args.morph_dir = "/home/ubuntu/volume/data/PostProcess_Data/digital/morph/after"
    if args.morphtype != "post_process" and morph != "post_process":
        args.morph_dir = args.root_dir
    wrapper = DatasetWrapper(args.root_dir, args.morphtype, morph_dir=args.morph_dir)
    trainds = wrapper.get_train_dataset(
        0, args.batch_size, morph_type=morph, shuffle=True, num_workers=8
    )
    testds = wrapper.get_test_dataset(
        0, args.batch_size, morph_type=morph, shuffle=True, num_workers=8
    )
    process_num = args.process_num
    print(morph)

    dir_path = "./data/print_scan_digital/embeddings"

    os.makedirs(dir_path, exist_ok=True)
    save_single_teacher_embds(
        morph, dir_path, trainds, "train", args.morphtype, logger, process_num
    )
    save_single_teacher_embds(
        morph, dir_path, testds, "test", args.morphtype, logger, process_num + 1
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    save(args)
