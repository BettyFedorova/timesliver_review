import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from sktime.datasets import load_UCR_UEA_dataset
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV

# =========================================================================
# 1. АРХИТЕКТУРА МОДЕЛИ TIMESLIVER (PyTorch)
# =========================================================================
class TimeSliver(nn.Module):
    def __init__(self, num_channels, seq_len, num_classes, num_bins=15, latent_dim=36, segment_size=7):
        super(TimeSliver, self).__init__()
        self.num_bins = num_bins
        self.latent_dim = latent_dim
        self.segment_size = segment_size
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.kappa = seq_len - segment_size + 1
        
        self.module_1_cnn = nn.Conv1d(num_channels, latent_dim, kernel_size=segment_size, stride=1)
        self.total_features = (num_bins * num_channels) * latent_dim
        self.fcls = nn.Linear(self.total_features, num_classes)

    def _compute_deterministic_Z(self, x):
        batch_size, seq_len, channels = x.shape
        device = x.device
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        
        bins = torch.linspace(-2.0, 2.0, self.num_bins - 1).to(device)
        x_discretized = torch.bucketize(x_norm, bins)
        z_one_hot = nn.functional.one_hot(x_discretized, num_classes=self.num_bins).float()
        
        z_one_hot = z_one_hot.view(batch_size, seq_len, -1).transpose(1, 2)
        avg_pool = nn.AvgPool1d(kernel_size=self.segment_size, stride=1)
        Z = avg_pool(z_one_hot)
        return Z.transpose(1, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x_transposed = x.transpose(1, 2)
        Q = self.module_1_cnn(x_transposed).transpose(1, 2)
        Z = self._compute_deterministic_Z(x)
        P = torch.bmm(Z.transpose(1, 2), Q)
        
        self.current_Z = Z
        self.current_Q = Q
        return self.fcls(P.view(batch_size, -1))

    def calculate_attribution(self, class_idx):
        weight_matrix = self.fcls.weight[class_idx].view(self.num_bins * self.num_channels, self.latent_dim)
        g_ij = weight_matrix.unsqueeze(0)
        sigma_ij = torch.sign(g_ij)
        
        Z_expanded = self.current_Z.unsqueeze(-1)
        Q_expanded = self.current_Q.unsqueeze(2)
        interaction = Z_expanded * Q_expanded
        
        signed_interaction = sigma_ij.unsqueeze(1) * interaction
        activated = torch.relu(signed_interaction)
        zeta_plus = torch.abs(g_ij).unsqueeze(1) * activated
        return zeta_plus.sum(dim=(2, 3))

# =========================================================================
# 2. ПОДГОТОВКА ДАННЫХ
# =========================================================================
print("Загрузка датасета BasicMotions из архива UCR/UEA...")
X_train_raw, y_train_raw = load_UCR_UEA_dataset(name="BasicMotions", split="train", return_type="numpy3d")
X_test_raw, y_test_raw = load_UCR_UEA_dataset(name="BasicMotions", split="test", return_type="numpy3d")

X_train_ts = np.transpose(X_train_raw, (0, 2, 1)).astype(np.float32)
X_test_ts = np.transpose(X_test_raw, (0, 2, 1)).astype(np.float32)

unique_labels = np.unique(y_train_raw)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train_ts = np.array([label_map[l] for l in y_train_raw], dtype=np.int64)
y_test_ts = np.array([label_map[l] for l in y_test_raw], dtype=np.int64)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_ts), torch.tensor(y_train_ts)), batch_size=8, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test_ts), torch.tensor(y_test_ts)), batch_size=8, shuffle=False)

num_samples, seq_len, num_channels = X_train_ts.shape
num_classes = len(unique_labels)

# =========================================================================
# 3. ОБУЧЕНИЕ TIMESLIVER И ПОЛУЧЕНИЕ КАРТЫ ОБЪЯСНЕНИЙ
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts_model = TimeSliver(num_channels=num_channels, seq_len=seq_len, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ts_model.parameters(), lr=0.002)

print("\n--- Обучение модели TimeSliver (PyTorch) ---")
start_time = time.time()
ts_model.train()
for epoch in range(1, 15):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(ts_model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
ts_duration = time.time() - start_time

ts_model.eval()
ts_correct = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        preds = torch.argmax(ts_model(batch_X), dim=1)
        ts_correct += (preds == batch_y).sum().item()
ts_accuracy = (ts_correct / len(X_test_ts)) * 100

# Генерируем карту объяснений (атрибуции) для первого тестового примера
sample_idx = 0
single_sample = torch.tensor(X_test_ts[sample_idx]).unsqueeze(0).to(device)
with torch.no_grad():
    pred_logits = ts_model(single_sample)
    pred_class = torch.argmax(pred_logits, dim=1).item()
    phi_map = ts_model.calculate_attribution(pred_class).squeeze(0).detach().cpu().numpy()


# =========================================================================
# 4. ОБУЧЕНИЕ baseline: ROCKET
# =========================================================================
print("\n--- Обучение классического baseline: ROCKET ---")
start_time = time.time()
rocket = Rocket(num_kernels=10000, random_state=42)
X_train_transformed = rocket.fit_transform(X_train_raw)
X_test_transformed = rocket.transform(X_test_raw)

rocket_classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
rocket_classifier.fit(X_train_transformed, y_train_raw)
rocket_duration = time.time() - start_time
rocket_accuracy = rocket_classifier.score(X_test_transformed, y_test_raw) * 100

# =========================================================================
# 5. СОХРАНЕНИЕ ГРАФИКОВ НА ДИСК
# =========================================================================
print("\nСохранение графиков результатов...")

# График 1: Сравнение метрик моделей
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
models = ['ROCKET', 'TimeSliver']
ax1.bar(models, [rocket_accuracy, ts_accuracy], color=['#34495e', '#e67e22'], width=0.5)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Сравнение точности моделей')
ax1.grid(axis='y', linestyle='--')

ax2.bar(models, [rocket_duration, ts_duration], color=['#34495e', '#2ecc71'], width=0.5)
ax2.set_ylabel('Время работы (сек)')
ax2.set_title('Сравнение времени обучения')
ax2.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('timesliver_vs_rocket_bar.png', dpi=150)
plt.close()

# График 2: Карта встроенной пошаговой важности (XAI)
plt.figure(figsize=(10, 4))
# Рисуем первый канал (сигнал датчика) тестового ряда
plt.plot(X_test_ts[sample_idx, :, 0], color='#1f77b4', label='Сигнал (Канал 1)', lw=2)
# Подсвечиваем цветом важность сегментов (phi_map растягиваем на длину ряда)
phi_resized = np.interp(np.linspace(0, len(phi_map), seq_len), np.arange(len(phi_map)), phi_map)
plt.imshow(phi_resized[np.newaxis, :], cmap='Oranges', aspect='auto', alpha=0.4, 
           extent=[0, seq_len, plt.ylim()[0], plt.ylim()[1]])
plt.colorbar(label='Интенсивность вклада f_att (Важность паттерна)')
plt.title(f'Встроенная карта объяснений TimeSliver (Предсказан класс: {pred_class})')
plt.xlabel('Временные шаги (Time Steps)')
plt.legend()
plt.tight_layout()
plt.savefig('timesliver_explanation_map.png', dpi=150)
plt.close()

# Вывод итоговой таблицы в консоль
print("\n" + "="*55)
print(f"{'МЕТОД / АЛГОРИТМ':<25} | {'ACCURACY (%)':<12} | {'ВРЕМЯ (сек)':<10}")
print("="*55)
print(f"{'ROCKET (Baseline)':<25} | {rocket_accuracy:<12.2f} | {rocket_duration:<10.3f}")
print(f"{'TimeSliver (ICLR 2026)':<25} | {ts_accuracy:<12.2f} | {ts_duration:<10.3f}")
print("="*55)
print("Все графики сохранены в текущую папку под именами:")
print(" - 'timesliver_vs_rocket_bar.png'")
print(" - 'timesliver_explanation_map.png'")
