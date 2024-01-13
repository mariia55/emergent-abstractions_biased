import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_loss = [
    0.6936,
    0.6348,
    0.5655,
    0.4938,
    0.3809,
    0.3210,
    0.2891,
    0.2729,
    0.2638,
    0.2502,
]
test_loss = [
    0.6902,
    0.6053,
    0.5805,
    0.4696,
    0.4026,
    0.3862,
    0.3857,
    0.3909,
    0.3528,
    0.3247,
]
train_accuracy = [
    0.5242,
    0.6351,
    0.6962,
    0.7732,
    0.8363,
    0.8649,
    0.8791,
    0.8864,
    0.8934,
    0.8984,
]
test_accuracy = [
    0.5028,
    0.6562,
    0.7315,
    0.7946,
    0.8455,
    0.8321,
    0.8495,
    0.8547,
    0.8753,
    0.8788,
]

# Create subplots for loss and accuracy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot loss curves
ax1.plot(epochs, train_loss, label="Train Loss", marker="o")
ax1.plot(epochs, test_loss, label="Test Loss", marker="o")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

# Plot accuracy curves
ax2.plot(epochs, train_accuracy, label="Train Accuracy", marker="o")
ax2.plot(epochs, test_accuracy, label="Test Accuracy", marker="o")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.legend()

# Display the plot
plt.tight_layout()
plt.show()
