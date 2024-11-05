import matplotlib.pyplot as plt

train_loss = [
    480.6995189189911,
    478.63749170303345,
    481.04301619529724,
    480.7931697368622,
    480.7200427055359,
    478.79445481300354,
    478.9083180427551,
    480.46302342414856,
    478.9014182090759,
    481.3308525085449,
    478.6679244041443,
    480.1141448020935,
]


num_epochs = len(train_loss)

fig, ax = plt.subplots()
ax.plot(range(num_epochs), train_loss, color="r", label="train")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss per Epoch")
plt.show()

plt.savefig("adapter_loss_analysis3.0.png")
