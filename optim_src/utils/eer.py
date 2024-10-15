import torch
import numpy as np
from tqdm import tqdm
import pickle
from torch.nn.functional import cosine_similarity
import os

from datasets.datawrapper import DatasetWrapper


# https://stackoverflow.com/questions/65230811/correct-compute-of-equal-error-rate-value


def calculate_eer(genuine, imposter, bins=10_001):
    genuine = np.squeeze(np.array(genuine))
    imposter = np.squeeze(np.array(imposter))
    far = np.ones(bins)
    frr = np.ones(bins)
    mi = np.min(imposter)
    mx = np.max(genuine)
    thresholds = np.linspace(mi, mx, bins)
    for id, threshold in enumerate(thresholds):
        fr = np.where(genuine <= threshold)[0].shape[0]
        fa = np.where(imposter >= threshold)[0].shape[0]
        frr[id] = fr * 100 / genuine.shape[0]
        far[id] = fa * 100 / imposter.shape[0]

    di = np.argmin(np.abs(far - frr))

    one = np.argmin(np.abs(far - 1))
    pointone = np.argmin(np.abs(far - 0.1))
    pointzeroone = np.argmin(np.abs(far - 0.01))
    pointzerozeroone = np.argmin(np.abs(far - 0.001))
    eer = (far[di] + frr[di]) / 2
    return (
        round(eer, 2),
        round(100 - frr[one], 2),
        round(100 - frr[pointone], 2),
        round(100 - frr[pointzeroone], 2),
        round(100 - frr[pointzerozeroone], 2),
    )


def classify(enrollfeat: torch.Tensor, probefeat: torch.Tensor) -> torch.Tensor:
    # return cosine_similarity(enrollfeat, probefeat, dim=1).cpu()
    return cosine_similarity(enrollfeat.unsqueeze(1), probefeat.unsqueeze(0), dim=2)


def eval_and_get_eer(model_name, model, morph, device, args, saving_dir, logger):
    if morph == "post_process":
        args.morph_dir = "/home/ubuntu/volume/data/PostProcess_Data/digital/morph/after"
    if morph != "post_process":
        args.morph_dir = args.root_dir
    wrapper = DatasetWrapper(args.root_dir, morph_type=morph, morph_dir=args.morph_dir)
    trainds = wrapper.get_train_dataset(
        0, args.batch_size, morph_type=morph, shuffle=True, num_workers=8
    )
    testds = wrapper.get_test_dataset(
        0, args.batch_size, morph_type=morph, shuffle=True, num_workers=8
    )
    model.to(device)
    model.eval()

    logger.debug("fetching scores for: %s", model_name)
    eer = None
    genuine_scores = []
    imposter_scores = []
    if args.isferet and os.path.exists(f"logs/scores/{model_name}/genuine_{model_name}_{morph}_feret.npy"):
        genuine_scores = np.load(
            f"logs/scores/{model_name}/genuine_{model_name}_{morph}_feret.npy"
        )
        imposter_scores = np.load(
            f"logs/scores/{model_name}/imposter_{model_name}_{morph}_feret.npy"
        )
    elif not args.isferet and os.path.exists(
        f"logs/scores/{model_name}/genuine_{model_name}_{morph}.npy"
    ):
        genuine_scores = np.load(
            f"logs/scores/{model_name}/genuine_{model_name}_{morph}.npy"
        )
        imposter_scores = np.load(
            f"logs/scores/{model_name}/imposter_{model_name}_{morph}.npy"
        )
    else:
        with torch.no_grad():
            enroll_features_list = []
            enroll_labels_list = []
            for img_path, enroll_batch, enroll_labels in tqdm(
                trainds, position=args.process_num
            ):
                enroll_batch = enroll_batch.to(device)
                enroll_features, embds = model(enroll_batch)
                enroll_features_list.append(enroll_features.cpu())
                enroll_labels = torch.argmax(enroll_labels, dim=1)
                enroll_labels_list.append(enroll_labels.cpu())

            enroll_features = torch.cat(enroll_features_list)
            enroll_labels = torch.cat(enroll_labels_list)

            for img_path, probe_batch, probe_labels in tqdm(testds):
                probe_batch = probe_batch.to(device)
                probe_features, embds = model(probe_batch)
                probe_labels = torch.argmax(probe_labels, dim=1)

                similarity_scores = classify(enroll_features.cuda(), probe_features)
                for i, probe_label in enumerate(probe_labels):
                    probe_label = probe_label.item()

                    scores_for_probe = similarity_scores[:, i]

                    genuine_mask = enroll_labels == probe_label
                    imposter_mask = ~genuine_mask

                    genuine_scores.extend(scores_for_probe[genuine_mask].tolist())
                    imposter_scores.extend(scores_for_probe[imposter_mask].tolist())

        os.makedirs(
            f"logs/scores/{model_name}",
            exist_ok=True,
        )
        if args.isferet:
            np.save(
                f"logs/scores/{model_name}/genuine_{model_name}_{morph}_feret.npy",
                genuine_scores,
            )
            np.save(
                f"logs/scores/{model_name}/imposter_{model_name}_{morph}_feret.npy",
                imposter_scores,
            )
        else:
            np.save(
                f"logs/scores/{model_name}/genuine_{model_name}_{morph}.npy",
                genuine_scores,
            )
            np.save(
                f"logs/scores/{model_name}/imposter_{model_name}_{morph}.npy",
                imposter_scores,
            )
    print(model_name, len(imposter_scores), len(genuine_scores), morph)
    eer, *_ = calculate_eer(genuine_scores, imposter_scores)

    return eer


def compute_eer(eval_morphs, models, device, saving_dir, args, logger) -> None:
    eer_table = {}
    for model_name, model in models.items():
        for morph in eval_morphs:
            if model_name not in eer_table:
                eer_table[model_name] = {}

            eer_table[model_name][morph] = eval_and_get_eer(
                model_name=model_name,
                model=model,
                morph=morph,
                device=device,
                args=args,
                saving_dir=saving_dir,
                logger=logger,
            )

    logger.info("EER Table:")
    for model_name, eers in eer_table.items():
        logger.info(f"{model_name}: {eers}")
        print(f"{model_name}: {eers}")
    print(saving_dir)
    if args.istest:
        if args.isferet:
            with open(f"{saving_dir}/eer_testdb_feret.pkl", "wb") as file:
                pickle.dump(eer_table, file)
        else:
            with open(f"{saving_dir}/eer_testdb.pkl", "wb") as file:
                pickle.dump(eer_table, file)
    else:
        with open(f"{saving_dir}/eer_traindb.pkl", "wb") as file:
            pickle.dump(eer_table, file)

    return
