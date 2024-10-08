import pickle
from matplotlib import legend
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot_hists(istest, saving_dir, models):
    # models = ["Teacher1", "Teacher2", "Teacher3", "Baseline", "Student"]
    eval_morphs = ["mipgan1", "cvmi", "lma", "stylegan"]
    saved_morphs = ["lmaubo", "mipgan2", "mordiff", "pipe"]
    if not istest:
        eval_morphs = saved_morphs

    saving_dir = saving_dir
    dir_path = f"{saving_dir}/score_charts"

    for model in models:
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
        for i, morph in enumerate(eval_morphs):
            data_bona = np.load(
                f"logs/scores/{model}/genuine_{model}_{morph}.npy",
                allow_pickle=True,
            )
            data_morph = np.load(
                f"logs/scores/{model}/imposter_{model}_{morph}.npy",
                allow_pickle=True,
            )
            # exp_plot(data_bona, data_morph, model)
            # axes[i].hist(
            #     [data_bona, data_morph],
            #     bins=50,
            #     stacked=True,
            #     color=["cyan", "Purple"],
            #     edgecolor="black",
            #     alpha=0.2,
            # )
            axes[i].hist(data_bona, alpha=0.5, label="bonafide", bins=50)
            axes[i].hist(data_morph, alpha=0.5, label="morph", bins=50)
            axes[i].set_title(morph)
            axes[i].legend(["bonafide", "morph"])

        plt.tight_layout()

        os.makedirs(f"{dir_path}", exist_ok=True)
        if istest:
            plt.savefig(f"{dir_path}/{model}_testdb.png")
        else:
            plt.savefig(f"{dir_path}/{model}_traindb.png")


def plot_charts(accuracy, test_loss, train_loss, plot_epoch, args, dir_path):
    num_epochs = plot_epoch + 1

    fig, ax = plt.subplots()
    ax.plot(range(num_epochs), train_loss, color="r", label="train")
    ax.plot(range(num_epochs), test_loss, color="g", label="test")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per Epoch")
    plt.show()

    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(
        f"{dir_path}/{args.model_name}_loss_{args.learning_rate}_{args.morphtype}_{args.eval_number}.png"
    )

    fig, ax = plt.subplots()
    ax.plot(range(num_epochs), accuracy)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("accuracy_list")
    ax.set_title("accuracy_list per Epoch")
    plt.show()
    plt.savefig(
        f"{dir_path}/{args.model_name}_acc1_{args.learning_rate}_{args.morphtype}_{args.eval_number}.png"
    )


def plot_combined_hist(istest, saving_dir, models) -> None:
    eval_morphs = ["mipgan1", "cvmi", "lma", "stylegan"]
    saved_morphs = ["lmaubo", "mipgan2", "mordiff", "pipe"]
    if not istest:
        eval_morphs = saved_morphs

    saving_dir = saving_dir
    dir_path = f"{saving_dir}/score_charts"

    for i, morph in enumerate(eval_morphs):
        plt.figure(figsize=(12, 6))
        legends = []
        for model in models:
            data_bona = np.load(
                f"logs/scores/{model}/genuine_{model}_{morph}.npy",
                allow_pickle=True,
            )
            data_morph = np.load(
                f"logs/scores/{model}/imposter_{model}_{morph}.npy",
                allow_pickle=True,
            )
            plt.hist(data_bona, alpha=0.5, label=f"bonafide_{model}", bins=70)
            plt.hist(data_morph, alpha=0.5, label=f"morph_{model}", bins=70)
            legends.append(f"bonafide_{model}")
            legends.append(f"morph_{model}")

        plt.legend(legends)

        plt.tight_layout()

        dir_path = f"{saving_dir}/score_charts/combined"

        os.makedirs(f"{dir_path}", exist_ok=True)
        if istest:
            plt.savefig(f"{dir_path}/{morph}_testdb.png")
        else:
            plt.savefig(f"{dir_path}/{morph}_traindb.png")


def plot_eer_bars(istest, saving_dir) -> None:
    if istest:
        eer_file = f"{saving_dir}/eer_testdb.pkl"
    else:
        eer_file = f"{saving_dir}/eer_traindb.pkl"

    with open(eer_file, "rb") as handle:
        eer_table = pickle.load(handle)

    df = pd.DataFrame(
        eer_table
    )  # Transpose to make keys as rows and target models as columns

    # Plotting a line chart
    # df.plot(kind="line", marker="o", figsize=(10, 6))

    # df.set_index("models", inplace=True)

    # Initialize the figure
    plt.figure(figsize=(10, 6))

    # Plot each column with lines and circular markers
    for column in df.columns:
        plt.plot(df.index, df[column], label=column, marker="o", markersize=5)

        # Highlight points with triangles if the model matches the column
        for model in df.index:
            if f"teacher_{model}" == column:  # Change condition as necessary
                plt.plot(
                    model, df[column][model], marker="^", markersize=8, color="red"
                )  # Triangle marker

    plt.title("EER Values")
    plt.xlabel("Teachers/Students")
    plt.ylabel("EER Value")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.legend(title="Morph types")
    plt.grid(True)  # Add grid lines to the plot
    plt.tight_layout()  # Adjust layout so that everything fits without overlap
    plt.show()
    dir_path = f"{saving_dir}/score_charts/eer"

    os.makedirs(f"{dir_path}", exist_ok=True)
    if istest:
        plt.savefig(f"{dir_path}/test_eer.png")
    else:
        plt.savefig(f"{dir_path}/train__eer.png")

    # target_models = list(next(iter(eer_table.values())).keys())

    # for target_model in target_models:
    #     plt.figure(figsize=(12, 6))

    #     eer_values = []
    #     labels = []

    #     for source, eers in eer_table.items():
    #         if target_model in eers:
    #             eer_values.append(eers[target_model])
    #             # Plot histogram for the current target model
    #             plt.hist(
    #                 eer_values, bins=50, alpha=0.5, label=source, edgecolor="black"
    #             )
    #             labels.append(source)

    #     # Add titles and labels
    #     plt.title(f"EER Histogram for {target_model}")
    #     plt.xlabel("EER")
    #     plt.ylabel("Frequency")
    #     plt.legend(labels)

    #     plt.tight_layout()

    # dir_path = f"{saving_dir}/score_charts/eer"

    # os.makedirs(f"{dir_path}", exist_ok=True)
    # if istest:
    #     plt.savefig(f"{dir_path}/{target_model}_eer.png")
    # else:
    #     plt.savefig(f"{dir_path}/{target_model}_eer.png")
