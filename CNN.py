import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # Saves figures without needing a GUI window
import numpy as np

# ─────────────────────────────────────────────
# 0. OUTPUT DIRECTORY
# ─────────────────────────────────────────────
os.makedirs("output", exist_ok=True)


# ─────────────────────────────────────────────
# 1. DATA PREPARATION
# ─────────────────────────────────────────────
# Data Augmentation (used ONLY during training)
# - RandomCrop: randomly crops 32x32 with a 4-pixel padding to preserve edges
# - RandomHorizontalFlip: randomly mirrors the image (50% probability)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test and validation transformation: normalization only, no augmentation
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training dataset twice, each with a different transformation
train_full_aug = CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
train_full_no_aug = CIFAR10(root='./data', train=True,  download=True, transform=test_transform)

test_data  = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Separate 10% of training data for validation (for early stopping)
num_train  = len(train_full_aug)           
val_size   = int(0.10 * num_train)         
train_size = num_train - val_size          

# Manually generate random indices to split training and validation
# This allows us to apply train_transform to the training set and test_transform to the validation set
indices = torch.randperm(num_train).tolist()
train_idx = indices[:train_size]
val_idx   = indices[train_size:]

train_data = Subset(train_full_aug, train_idx)
val_data   = Subset(train_full_no_aug, val_idx)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

print(f"Train     : {len(train_data)} images (With Data Augmentation)")
print(f"Validation: {len(val_data)} images (No Data Augmentation)")
print(f"Test      : {len(test_data)} images (No Data Augmentation)")

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# ─────────────────────────────────────────────
# 2. FLEXIBLE CNN CLASS
# ─────────────────────────────────────────────
# Why use a flexible class?
#   Instead of creating three completely separate architectures, we define
#   a single class that receives the configuration as a parameter.
#   This makes it easy to instantiate different topologies.

class CNN(nn.Module):
    """
    Flexible Convolutional Network for image classification.

    Parameters
    ----------
    conv_configs : list of dictionaries, each with:
                   - 'filters' : number of filters (output channels)
                   - 'kernel'  : filter size (e.g., 3 -> 3x3 filter)
                   Each entry creates a Conv -> ReLU -> MaxPool(2x2) layer
    fc_sizes     : list with the number of neurons for each hidden FC layer.
                   The final output layer (10 classes) is added automatically.
    dropout_p    : dropout probability in FC layers (0 = no dropout)
    """

    def __init__(self, conv_configs, fc_sizes, dropout_p=0.0):
        super(CNN, self).__init__()   

        # ── Convolutional Block ──────────────────────────────────────────
        conv_layers = []
        in_channels = 3   # CIFAR-10: RGB images -> 3 input channels

        for cfg in conv_configs:
            # padding=1 keeps spatial size (H and W) unchanged after convolution
            conv_layers.append(
                nn.Conv2d(in_channels, cfg['filters'],
                          kernel_size=cfg['kernel'], padding=cfg['kernel']//2)
            )
            conv_layers.append(nn.ReLU())               
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  

            in_channels = cfg['filters']   

        self.conv_block = nn.Sequential(*conv_layers)

        # ── Calculate vector size after all conv/pool layers ─────────────
        # CIFAR-10: 32x32. Each MaxPool(2x2) halves the spatial dimensions.
        # With N conv layers, final spatial size is 32 / 2^N.
        n_pools    = len(conv_configs)
        final_size = 32 // (2 ** n_pools)          
        flat_size  = in_channels * final_size * final_size   

        # ── Fully Connected (FC) Block ───────────────────────────────────
        fc_layers = []
        in_features = flat_size

        for out_features in fc_sizes:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            if dropout_p > 0:
                fc_layers.append(nn.Dropout(p=dropout_p))
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, 10))

        self.fc_block = nn.Sequential(*fc_layers)

    def forward(self, x):
        """Forward pass."""
        x = self.conv_block(x)        
        x = x.view(x.size(0), -1)     
        x = self.fc_block(x)          
        return x                      


# ─────────────────────────────────────────────
# 3. ARCHITECTURE DEFINITIONS
# ─────────────────────────────────────────────
# We vary:
#   - Number of conv layers (2, 3, or 4)
#   - Number of filters (32, 64, 128)
#   - Dropout (0.0, 0.1, 0.3)
# This allows us to compare the impact of each design choice.

architectures = {

    # Architecture 1: SIMPLE — 2 conv layers, smaller filters, no dropout
    # Serves as a baseline to compare against the others.
    "CNN_A": {
        "conv_configs": [
            {"filters": 32, "kernel": 3},   # conv1: 32 filters 3x3
            {"filters": 64, "kernel": 3},   # conv2: 64 filters 3x3
        ],
        "fc_sizes"   : [256, 128],
        "dropout_p"  : 0.0,
        "optimizer"  : "adam",
        "lr"         : 1e-3,
    },

    # Architecture 2: MEDIUM — 3 conv layers, larger initial kernel, light dropout
    "CNN_B": {
        "conv_configs": [
            {"filters": 64,  "kernel": 5},  # conv1: 64 filters 5x5
            {"filters": 128, "kernel": 3},  # conv2: 128 filters 3x3
            {"filters": 128, "kernel": 3},  # conv3: 128 filters 3x3
        ],
        "fc_sizes"   : [200, 200],          # FC1(200) -> FC2(200) -> FC3(10)
        "dropout_p"  : 0.1,
        "optimizer"  : "adam",
        "lr"         : 1e-3,
    },

    # Architecture 3: DEEP — 4 conv layers, larger filters, higher dropout
    # More capacity, but higher risk of overfitting -> testing dropout=0.3
    "CNN_C": {
        "conv_configs": [
            {"filters": 32,  "kernel": 3},
            {"filters": 64,  "kernel": 3},
            {"filters": 128, "kernel": 3},
            {"filters": 128, "kernel": 3},
        ],
        "fc_sizes"   : [256, 128],
        "dropout_p"  : 0.3,
        "optimizer"  : "sgd",
        "lr"         : 1e-2,   # SGD generally requires a higher lr than Adam
    },
}


# ─────────────────────────────────────────────
# 4. TRAINING AND EVALUATION FUNCTIONS
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Trains the model for one full epoch. Returns avg loss and accuracy."""
    model.train()   
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          
        outputs = model(images)        
        loss    = criterion(outputs, labels)   
        loss.backward()                
        optimizer.step()               

        total_loss += loss.item()
        _, predicted = outputs.max(1)  
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluates the model without updating weights. Returns avg loss and accuracy."""
    model.eval()   
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():   
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(name, config, train_loader, val_loader, device,
                num_epochs=20, patience=5):
    """
    Trains a model using early stopping based on validation loss.

    Parameters
    ----------
    patience : number of epochs with no improvement before stopping the training.
    """
    print(f"\n{'='*55}")
    print(f"  Training: {name}")
    print(f"{'='*55}")

    model = CNN(config["conv_configs"], config["fc_sizes"], config["dropout_p"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    # Early stopping: saves the best model state and counts epochs without improvement
    best_val_loss  = float('inf')
    epochs_no_impr = 0
    best_state     = None

    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"  Epoch {epoch:02d}/{num_epochs} | "
              f"Train: loss={t_loss:.4f} acc={t_acc:.1f}% | "
              f"Val:   loss={v_loss:.4f} acc={v_acc:.1f}%")

        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            epochs_no_impr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                print(f"  -> Early stopping triggered at epoch {epoch}.")
                break

    # Restore weights of the best model found during training
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ─────────────────────────────────────────────
# 5. TRAINING PLOTS
# ─────────────────────────────────────────────

def plot_history(name, history):
    """Plots training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training History - {name}", fontsize=13)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Validation")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"output/{name}_history.png", dpi=100)
    plt.close()
    print(f"  [Saved] output/{name}_history.png")


# ─────────────────────────────────────────────
# 6. FILTER VISUALIZATION
# ─────────────────────────────────────────────

def plot_filters(model, name):
    """
    Visualizes the filters of the FIRST convolutional layer.
    Averages the 3 RGB channels to convert to grayscale for easier visualization.
    """
    # Retrieves weights from the first Conv2d layer
    first_conv = None
    for layer in model.conv_block:
        if isinstance(layer, nn.Conv2d):
            first_conv = layer
            break

    if first_conv is None:
        return

    weights = first_conv.weight.data.cpu()

    num_filters = min(16, weights.shape[0])   
    cols = 8
    rows = (num_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"1st Conv Layer Filters - {name}", fontsize=12)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis('off')
        if i < num_filters:
            # Average of 3 channels -> Grayscale
            filt = weights[i].mean(dim=0).numpy()
            # Normalize to [0, 1] for visualization
            filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
            ax.imshow(filt, cmap='viridis')

    plt.tight_layout()
    plt.savefig(f"output/{name}_filters.png", dpi=100)
    plt.close()
    print(f"  [Saved] output/{name}_filters.png")


# ─────────────────────────────────────────────
# 7. ACTIVATION MAP VISUALIZATION
# ─────────────────────────────────────────────

def plot_activation_maps(model, loader, name, device):
    """
    Passes an image through the conv block and displays activation maps
    from the LAST convolutional layer.
    """
    model.eval()

    images, labels = next(iter(loader))
    img = images[0:1].to(device)   

    # Pass the image exclusively through the convolutional block
    with torch.no_grad():
        feat_maps = model.conv_block(img)   

    feat_maps = feat_maps.squeeze(0).cpu()  

    num_maps = min(16, feat_maps.shape[0])
    cols = 8
    rows = (num_maps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"Activation Maps (last conv) - {name}", fontsize=12)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis('off')
        if i < num_maps:
            fmap = feat_maps[i].numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap='inferno')

    plt.tight_layout()
    plt.savefig(f"output/{name}_activation_maps.png", dpi=100)
    plt.close()
    print(f"  [Saved] output/{name}_activation_maps.png")


# ─────────────────────────────────────────────
# 8. PREDICTED CLASS IMAGES
# ─────────────────────────────────────────────

def plot_predictions(model, loader, name, device, n=16):
    """
    Displays n images from test_loader with their predicted and true classes.
    Green = correct, Red = incorrect.
    """
    model.eval()
    images, labels = next(iter(loader))
    images_dev = images[:n].to(device)

    with torch.no_grad():
        outputs = model(images_dev)
        _, preds = outputs.max(1)

    preds  = preds.cpu().numpy()
    labels = labels[:n].numpy()
    images = images[:n]

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    fig.suptitle(f"Test Predictions - {name}", fontsize=12)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis('off')
        if i < n:
            # Revert normalization to display the image correctly
            img = images[i].numpy().transpose(1, 2, 0)  
            img = img * 0.5 + 0.5                        
            img = np.clip(img, 0, 1)
            ax.imshow(img)

            correct = (preds[i] == labels[i])
            color   = 'green' if correct else 'red'
            ax.set_title(f"P:{CLASS_NAMES[preds[i]]}\nT:{CLASS_NAMES[labels[i]]}",
                         fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(f"output/{name}_predictions.png", dpi=100)
    plt.close()
    print(f"  [Saved] output/{name}_predictions.png")


# ─────────────────────────────────────────────
# 9. FINAL COMPARISON
# ─────────────────────────────────────────────

def plot_comparison(results):
    """Bar chart comparing test accuracy across architectures."""
    names = list(results.keys())
    accs  = [results[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, accs, color=['steelblue', 'darkorange', 'seagreen'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("CNN Architectures Comparison - CIFAR-10")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%", ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig("output/architectures_comparison.png", dpi=100)
    plt.close()
    print("\n[Saved] output/architectures_comparison.png")


# ─────────────────────────────────────────────
# 10. MAIN LOOP
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    results = {}   
    log_lines = [] 

    for name, config in architectures.items():

        model, history = train_model(
            name, config, train_loader, val_loader, device,
            num_epochs=50, patience=5
        )

        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        results[name] = test_acc
        print(f"\n  -> TEST Accuracy ({name}): {test_acc:.2f}%\n")

        log_lines.append(f"{name}: test_acc={test_acc:.2f}%, test_loss={test_loss:.4f}")

        plot_history(model_name := name, history)
        plot_filters(model, name)
        plot_activation_maps(model, test_loader, name, device)
        plot_predictions(model, test_loader, name, device)

    plot_comparison(results)

    with open("output/results.txt", "w") as f:
        f.write("Final Results - CIFAR-10\n")
        f.write("=" * 40 + "\n")
        for line in log_lines:
            f.write(line + "\n")

    print("\n[Saved] output/results.txt")
    print("\nAll done! Check the output/ folder.")


if __name__ == "__main__":
    main()