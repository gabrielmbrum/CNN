import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # Salva figuras sem precisar de janela gráfica
import numpy as np

# ─────────────────────────────────────────────
# 0. DIRETÓRIO DE SAÍDA
# ─────────────────────────────────────────────
os.makedirs("output", exist_ok=True)


# ─────────────────────────────────────────────
# 1. PREPARAÇÃO DOS DADOS
# ─────────────────────────────────────────────
# Data Augmentation (utilizado APENAS no treino)
# - RandomCrop: recorta aleatoriamente 32x32 com um padding de 4 pixels (evita perder bordas vitais)
# - RandomHorizontalFlip: espelha a imagem aleatoriamente (50% de chance)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Transformação para teste e validação: apenas normalização, sem aumento de dados
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Carregamos o dataset de treino duas vezes, cada um com uma transformação diferente
train_full_aug = CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
train_full_no_aug = CIFAR10(root='./data', train=True,  download=True, transform=test_transform)

# Carregamos o teste normalmente
test_data  = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Separamos 10% do treino como validação (para early stopping)
num_train  = len(train_full_aug)           # 50000
val_size   = int(0.10 * num_train)         # 5000 imagens
train_size = num_train - val_size          # 45000 imagens

# Geramos índices aleatórios para separar treino e validação
# Fazemos isso manualmente em vez do random_split para podermos aplicar 
# o train_transform no treino e o test_transform na validação
indices = torch.randperm(num_train).tolist()
train_idx = indices[:train_size]
val_idx   = indices[train_size:]

# Criamos os Subsets puxando dos datasets instanciados corretamente
train_data = Subset(train_full_aug, train_idx)
val_data   = Subset(train_full_no_aug, val_idx)

# DataLoaders: iteram sobre os dados em mini-batches
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

print(f"Treino   : {len(train_data)} imagens (Com Data Augmentation)")
print(f"Validação: {len(val_data)} imagens (Sem Data Augmentation)")
print(f"Teste    : {len(test_data)} imagens (Sem Data Augmentation)")

# Nomes das 10 classes do CIFAR-10
CLASS_NAMES = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo',
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']


# ─────────────────────────────────────────────
# 2. CLASSE CNN FLEXÍVEL
# ─────────────────────────────────────────────
# Por que usar uma classe flexível?
#   Em vez de criar três arquiteturas completamente separadas, definimos
#   uma única classe que recebe a configuração como parâmetro.
#   Assim podemos instanciar diferentes topologias facilmente.

class CNN(nn.Module):
    """
    Rede Convolucional Flexível para classificação de imagens.

    Parâmetros
    ----------
    conv_configs : lista de dicionários, cada um com:
                   - 'filters' : número de filtros (canais de saída)
                   - 'kernel'  : tamanho do filtro (ex: 3 → filtro 3x3)
                   Cada entrada cria uma camada Conv → ReLU → MaxPool(2x2)
    fc_sizes     : lista com o número de neurônios de cada camada FC oculta.
                   A última camada de saída (10 classes) é adicionada automaticamente.
    dropout_p    : probabilidade de dropout nas camadas FC (0 = sem dropout)
    """

    def __init__(self, conv_configs, fc_sizes, dropout_p=0.0):
        super(CNN, self).__init__()   # inicializa a classe pai nn.Module

        # ── Bloco convolucional ──────────────────────────────────────────
        # Construímos as camadas convolucionais dinamicamente.
        conv_layers = []
        in_channels = 3   # CIFAR-10: imagens RGB → 3 canais de entrada

        for cfg in conv_configs:
            # Camada convolucional:
            #   in_channels → cfg['filters'] mapas de características
            #   padding=1 mantém o tamanho espacial (H e W) inalterado após a conv
            conv_layers.append(
                nn.Conv2d(in_channels, cfg['filters'],
                          kernel_size=cfg['kernel'], padding=cfg['kernel']//2)
            )
            conv_layers.append(nn.ReLU())               # ativação não-linear
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # reduz H e W pela metade

            in_channels = cfg['filters']   # próxima conv recebe os filtros desta como entrada

        # nn.Sequential empacota a lista como uma única "caixa" sequencial
        self.conv_block = nn.Sequential(*conv_layers)

        # ── Calcular o tamanho do vetor após todas as convs/poolings ────
        # CIFAR-10: 32x32. Cada MaxPool(2x2) divide por 2.
        # Com N camadas conv, o tamanho espacial final é 32 / 2^N.
        n_pools    = len(conv_configs)
        final_size = 32 // (2 ** n_pools)          # tamanho H (= W) após os pools
        flat_size  = in_channels * final_size * final_size   # total de valores achatados

        # ── Bloco Fully Connected (FC) ───────────────────────────────────
        fc_layers = []
        in_features = flat_size

        for out_features in fc_sizes:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            if dropout_p > 0:
                fc_layers.append(nn.Dropout(p=dropout_p))
            in_features = out_features

        # Camada de saída: 10 logits (um por classe)
        fc_layers.append(nn.Linear(in_features, 10))

        self.fc_block = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Passagem para frente (forward pass).
        x: tensor de forma (batch, 3, 32, 32)
        """
        x = self.conv_block(x)        # extrai características
        x = x.view(x.size(0), -1)     # "achata" para (batch, flat_size)
        x = self.fc_block(x)          # classificação
        return x                       # retorna logits (CrossEntropyLoss aplica softmax internamente)


# ─────────────────────────────────────────────
# 3. DEFINIÇÃO DAS 3 ARQUITETURAS
# ─────────────────────────────────────────────
# Variamos:
#   - Número de camadas convolucionais (2, 3 ou 4)
#   - Número de filtros (32, 64, 128)
#   - Dropout (0.0, 0.1, 0.3)
# Isso permite comparar o impacto de cada escolha.

architectures = {

    # Arquitetura 1: SIMPLES — 2 camadas conv, filtros menores, sem dropout
    # Serve como baseline (referência) para comparar com as demais.
    "CNN_A": {
        "conv_configs": [
            {"filters": 32, "kernel": 3},   # conv1: 32 filtros 3x3
            {"filters": 64, "kernel": 3},   # conv2: 64 filtros 3x3
        ],
        "fc_sizes"   : [256, 128],
        "dropout_p"  : 0.0,
        "optimizer"  : "adam",
        "lr"         : 1e-3,
    },

    # Arquitetura 2: MÉDIA — 3 camadas conv, inspirada no exemplo do enunciado, dropout leve
    "CNN_B": {
        "conv_configs": [
            {"filters": 64,  "kernel": 5},  # conv1: 64 filtros 5x5 (como no enunciado)
            {"filters": 128, "kernel": 3},  # conv2: 128 filtros 3x3
            {"filters": 128, "kernel": 3},  # conv3: 128 filtros 3x3
        ],
        "fc_sizes"   : [200, 200],          # FC1(200) → FC2(200) → FC3(10)
        "dropout_p"  : 0.1,
        "optimizer"  : "adam",
        "lr"         : 1e-3,
    },

    # Arquitetura 3: PROFUNDA — 4 camadas conv, filtros maiores, dropout mais alto
    # Mais capacidade, mas maior risco de overfitting → testamos dropout=0.3
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
        "lr"         : 1e-2,   # SGD geralmente precisa de lr maior que Adam
    },
}


# ─────────────────────────────────────────────
# 4. FUNÇÕES DE TREINAMENTO E AVALIAÇÃO
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Treina o modelo por uma época completa.
    Retorna a perda média e a acurácia da época.
    """
    model.train()   # modo treino: habilita dropout, batch norm etc.
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # limpa gradientes do passo anterior
        outputs = model(images)        # forward pass
        loss    = criterion(outputs, labels)   # calcula perda
        loss.backward()                # backward: calcula gradientes
        optimizer.step()               # atualiza pesos

        total_loss += loss.item()
        _, predicted = outputs.max(1)  # classe com maior logit
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Avalia o modelo (sem atualizar pesos).
    Retorna perda média e acurácia.
    """
    model.eval()   # modo avaliação: desativa dropout
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():   # desliga cálculo de gradientes (economiza memória)
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
    Treina um modelo com early stopping baseado na perda de validação.

    Parâmetros
    ----------
    patience : número de épocas sem melhora antes de parar o treino.
               Isso evita overfitting e economiza tempo.
    """
    print(f"\n{'='*55}")
    print(f"  Treinando: {name}")
    print(f"{'='*55}")

    model = CNN(config["conv_configs"], config["fc_sizes"], config["dropout_p"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # adequado para classificação multi-classe

    # Escolhe otimizador conforme configuração
    if config["optimizer"] == "adam":
        # Adam adapta o lr individualmente para cada peso → converge mais rápido
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        # SGD é mais simples e às vezes generaliza melhor com lr adequado
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    # Histórico para gerar gráficos depois
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    # Early stopping: guardamos o melhor modelo e contamos épocas sem melhora
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

        print(f"  Época {epoch:02d}/{num_epochs} | "
              f"Treino: loss={t_loss:.4f} acc={t_acc:.1f}% | "
              f"Val:   loss={v_loss:.4f} acc={v_acc:.1f}%")

        # Verifica se houve melhora na validação
        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            epochs_no_impr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                print(f"  → Early stopping acionado na época {epoch}.")
                break

    # Restaura os pesos do melhor modelo encontrado
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ─────────────────────────────────────────────
# 5. PLOTS DE TREINAMENTO
# ─────────────────────────────────────────────

def plot_history(name, history):
    """Plota curvas de perda e acurácia de treino/validação."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Histórico de Treinamento — {name}", fontsize=13)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Perda
    axes[0].plot(epochs, history["train_loss"], label="Treino")
    axes[0].plot(epochs, history["val_loss"],   label="Validação")
    axes[0].set_title("Perda (Loss)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].legend()

    # Acurácia
    axes[1].plot(epochs, history["train_acc"], label="Treino")
    axes[1].plot(epochs, history["val_acc"],   label="Validação")
    axes[1].set_title("Acurácia (%)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"output/{name}_historico.png", dpi=100)
    plt.close()
    print(f"  [Salvo] output/{name}_historico.png")


# ─────────────────────────────────────────────
# 6. VISUALIZAÇÃO DE FILTROS
# ─────────────────────────────────────────────

def plot_filters(model, name):
    """
    Visualiza os filtros da PRIMEIRA camada convolucional.
    Cada filtro tem forma (3, K, K) → plotamos os 3 canais RGB juntos.
    Para isso convertemos para escala de cinza fazendo a média dos canais.
    """
    # Pega os pesos da primeira Conv2d do bloco convolucional
    first_conv = None
    for layer in model.conv_block:
        if isinstance(layer, nn.Conv2d):
            first_conv = layer
            break

    if first_conv is None:
        return

    # weights shape: (num_filters, 3, K, K)
    weights = first_conv.weight.data.cpu()

    num_filters = min(16, weights.shape[0])   # mostra no máximo 16 filtros
    cols = 8
    rows = (num_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"Filtros da 1ª Camada Conv — {name}", fontsize=12)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis('off')
        if i < num_filters:
            # Média dos 3 canais → escala de cinza
            filt = weights[i].mean(dim=0).numpy()
            # Normaliza para [0, 1] para facilitar a visualização
            filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
            ax.imshow(filt, cmap='viridis')

    plt.tight_layout()
    plt.savefig(f"output/{name}_filtros.png", dpi=100)
    plt.close()
    print(f"  [Salvo] output/{name}_filtros.png")


# ─────────────────────────────────────────────
# 7. VISUALIZAÇÃO DE MAPAS DE ATIVAÇÃO
# ─────────────────────────────────────────────

def plot_activation_maps(model, loader, name, device):
    """
    Passa uma imagem pelo bloco convolucional e mostra os mapas de ativação
    (feature maps) da ÚLTIMA camada convolucional.
    """
    model.eval()

    # Pega um batch e usa apenas a primeira imagem
    images, labels = next(iter(loader))
    img = images[0:1].to(device)   # shape: (1, 3, 32, 32)

    # Passamos a imagem somente pelo bloco convolucional
    with torch.no_grad():
        feat_maps = model.conv_block(img)   # shape: (1, C, H, W)

    feat_maps = feat_maps.squeeze(0).cpu()  # → (C, H, W)

    num_maps = min(16, feat_maps.shape[0])
    cols = 8
    rows = (num_maps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"Mapas de Ativação (última conv) — {name}", fontsize=12)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis('off')
        if i < num_maps:
            fmap = feat_maps[i].numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap='inferno')

    plt.tight_layout()
    plt.savefig(f"output/{name}_mapas_ativacao.png", dpi=100)
    plt.close()
    print(f"  [Salvo] output/{name}_mapas_ativacao.png")


# ─────────────────────────────────────────────
# 8. IMAGENS COM CLASSES PREDITAS
# ─────────────────────────────────────────────

def plot_predictions(model, loader, name, device, n=16):
    """
    Mostra n imagens do test_loader com a classe predita e a real.
    Verde = acerto, Vermelho = erro.
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
    fig.suptitle(f"Predições no Teste — {name}", fontsize=12)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis('off')
        if i < n:
            # Desfaz a normalização para exibir a imagem corretamente
            img = images[i].numpy().transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
            img = img * 0.5 + 0.5                        # [-1,1] → [0,1]
            img = np.clip(img, 0, 1)
            ax.imshow(img)

            correct = (preds[i] == labels[i])
            color   = 'green' if correct else 'red'
            ax.set_title(f"P:{CLASS_NAMES[preds[i]]}\nR:{CLASS_NAMES[labels[i]]}",
                         fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(f"output/{name}_predicoes.png", dpi=100)
    plt.close()
    print(f"  [Salvo] output/{name}_predicoes.png")


# ─────────────────────────────────────────────
# 9. COMPARAÇÃO FINAL
# ─────────────────────────────────────────────

def plot_comparison(results):
    """Gráfico de barras com a acurácia no teste de cada arquitetura."""
    names = list(results.keys())
    accs  = [results[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, accs, color=['steelblue', 'darkorange', 'seagreen'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Acurácia no Teste (%)")
    ax.set_title("Comparação de Arquiteturas CNN — CIFAR-10")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%", ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig("output/comparacao_arquiteturas.png", dpi=100)
    plt.close()
    print("\n[Salvo] output/comparacao_arquiteturas.png")


# ─────────────────────────────────────────────
# 10. LOOP PRINCIPAL
# ─────────────────────────────────────────────

def main():
    # Detecta se há GPU disponível; senão usa CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    results = {}   # guarda acurácia no teste de cada arquitetura
    log_lines = [] # log de texto para salvar em arquivo

    for name, config in architectures.items():

        # Treina
        model, history = train_model(
            name, config, train_loader, val_loader, device,
            num_epochs=50, patience=5
        )

        # Avalia no teste
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        results[name] = test_acc
        print(f"\n  → Acurácia no TESTE ({name}): {test_acc:.2f}%\n")

        log_lines.append(f"{name}: teste_acc={test_acc:.2f}%, teste_loss={test_loss:.4f}")

        # Gera plots
        plot_history(model_name := name, history)
        plot_filters(model, name)
        plot_activation_maps(model, test_loader, name, device)
        plot_predictions(model, test_loader, name, device)

    # Gráfico comparativo
    plot_comparison(results)

    # Salva log em texto
    with open("output/resultados.txt", "w") as f:
        f.write("Resultados finais — CIFAR-10\n")
        f.write("=" * 40 + "\n")
        for line in log_lines:
            f.write(line + "\n")

    print("\n[Salvo] output/resultados.txt")
    print("\nTudo concluído! Verifique a pasta output/")


if __name__ == "__main__":
    main()