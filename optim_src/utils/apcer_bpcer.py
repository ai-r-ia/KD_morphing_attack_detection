import torch
from tqdm import tqdm
import pickle

from datasets.datawrapper import DatasetWrapper

# bonafide = 0, attack = 1 (check one hot encoding in datawrapper)


def calculate_apcer_bpcer(model, morph, device, args):
    wrapper = DatasetWrapper(args.root_dir, morph_type=morph)
    trainds = wrapper.get_train_dataset(
        2, args.batch_size, morph_type=args.morphtype, shuffle=True, num_workers=8
    )  # TODO: ??
    testds = wrapper.get_test_dataset(
        2, args.batch_size, morph_type=args.morphtype, shuffle=True, num_workers=8
    )

    dataloaders = [trainds, testds]
    model.to(device)
    model.eval()

    total_attacks = 0
    total_bonafide = 0

    correct_attacks = 0
    correct_bonafide = 0

    with torch.no_grad():
        for dataloader in dataloaders:
            for x, y in tqdm(dataloader):
                x, y = x.to(device), y.to(device)
                preds, embds = model(x)

                # preds output are logits, convert them to predicted y
                _, pred_class = torch.max(preds, dim=1)
                y = torch.argmax(y, dim=1)
                attack_mask = y == 1
                bonafide_mask = y == 0

                total_attacks += attack_mask.sum().item()
                total_bonafide += bonafide_mask.sum().item()

                # print(pred_class, attack_mask)
                correct_attacks += (pred_class[attack_mask] == 0).sum().item()
                correct_bonafide += (pred_class[bonafide_mask] == 1).sum().item()

                print(correct_attacks, total_attacks)
                print(correct_bonafide, total_bonafide)

    apcer = correct_attacks / total_attacks if total_attacks > 0 else 0.0
    bpcer = correct_bonafide / total_bonafide if total_bonafide > 0 else 0.0

    return apcer, bpcer


def compute_apcer_bpcer(eval_morphs, models, device, saving_dir, args):
    apcer_table = {}
    bpcer_table = {}

    for morph in eval_morphs:
        for model_name, model in models.items():
            if model_name not in apcer_table:
                apcer_table[model_name] = {}
                bpcer_table[model_name] = {}
            print(model_name)
            apcer_table[model_name][morph], bpcer_table[model_name][morph] = (
                calculate_apcer_bpcer(
                    model=model, morph=morph, device=device, args=args
                )
            )

    print("apcer Table:")
    for model_name, apcers in apcer_table.items():
        print(f"{model_name}: {apcers}")

    with open(f"{saving_dir}/apcer.pkl", "wb") as file:
        pickle.dump(apcer_table, file)

    print("bpcer Table:")
    for model_name, bpcers in bpcer_table.items():
        print(f"{model_name}: {bpcers}")

    with open(f"{saving_dir}/bpcer.pkl", "wb") as file:
        pickle.dump(bpcer_table, file)
    return
