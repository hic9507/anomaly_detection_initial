import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import cv2
import os
import glob

args = {"data_folder": "D:\\InChang\\abnormal_detection_dataset\\videos\\videos\\",
        "graphs_folder": "./graphs/UCF-CRIME/", "epoch": 50, "batch_size": 16, "num_class": 1, "learning_rate": 0.001,
        # 원래 1e-4
        "decay_rate": 0.001, "num_workers": 4, "img_size": (64, 64), "img_depth": 3}


# CBAM 모듈 정의
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class CustomDataset(Dataset):

    def __init__(self, dataset_folder_path=args["data_folder"], image_size=args["img_size"], image_depth=3, train=True,
                 transform=None):
        self.dataset_folder_path = dataset_folder_path
        self.transform = transform  # HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경
        self.image_size = image_size
        self.image_depth = image_depth
        self.train = train
        self.classes = sorted(self.get_classnames())
        self.image_path_label = self.read_folder()

    def __len__(self):
        # print(self.image_path_label)
        return len(self.image_path_label)

    def __getitem__(self, idx):
        frames = []
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, label = self.image_path_label[idx]
        cap = cv2.VideoCapture(image)
        fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # for frame in range(args["FPS"]):
        for frame in range(int(fps)):
            frame, image = cap.read()
            if self.image_depth == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.image_size)
                image = image / 255
            else:
                image = cv2.resize(image, self.image_size)
                image = image / 255

        return {'image': torch.FloatTensor(image).permute(2, 0, 1),
                'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장

    def get_classnames(self):
        # print('---------------------------')
        # print(os.listdir(f"{self.dataset_folder_path.rstrip('/')}/Train/"))
        return os.listdir(f"{self.dataset_folder_path}train/")  # violence, nonviolence

    def read_folder(self):
        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path}train/"
        else:
            folder_path = f"{self.dataset_folder_path}val/"  # valid도 가능

        for x in glob.glob(folder_path + "**", recursive=True):  # 해당 폴더의 하위 폴더까지 탐색
            if not x.endswith('mp4'):  # rwf2000, ubi-fights: mp4
                continue
            # print("x.split('\\')", x.split('\\'))
            class_idx = self.classes.index(x.split('\\')[-2])  # 클래스 이름(Violence, Nonviolence)의 인덱스 저장 # ubi, ucfcrime
            # print('class_idx: ', class_idx, ' ', 'classes.index(x.split("\\")[-2]: ', x.split('\\')[-2])
            # print("self.classes.index(x.split('\\')[-2]):", x.split('\\')[-2])
            image_path_label.append(
                (x, int(class_idx)))  # ('D:\\abnormal_detection_dataset\\RWF_2000_Dataset\\train\\Fight\\fi243.avi', 0)

        return image_path_label  # (이미지 경로, 클래스 인덱스)


##### 콜백함수
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, verbose=False, delta=0,
                 path=f"{model_save_folder}UBI-FIGHTS_640x360_videocapture_model_earlywithout_Cbam.pth"):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print('--------------------------------------------------------------------------------\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  val loss is decreased, Saving model ...')
            print('\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 하이퍼파라미터 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 및 데이터로더 설정
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = UbiFightsDataset(root='path/to/train/dataset', transform=transform)
valid_dataset = UbiFightsDataset(root='path/to/validation/dataset', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 모델 초기화 및 CBAM 적용
model = densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)
model.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.features.norm0 = nn.BatchNorm2d(64)
model.features = nn.Sequential(
    model.features,
    CBAM(1024)
)
model.to(device)

# 손실 함수 및 최적화 알고리즘 설정
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# 훈련 및 검증 함수
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predicted = torch.sigmoid(outputs) >= 0.5
        total += labels.size(0)
        correct += predicted.eq(labels.byte().unsqueeze(1)).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))

            running_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs) >= 0.5
            total += labels.size(0)
            correct += predicted.eq(labels.byte().unsqueeze(1)).sum().item()

            y_true.extend(labels.tolist())
            y_pred.extend(torch.sigmoid(outputs).tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total * 100
    epoch_auc = roc_auc_score(y_true, y_pred)
    return epoch_loss, epoch_acc, epoch_auc


# 학습 및 검증 실행
train_losses = []
train_accs = []
valid_losses = []
valid_accs = []
valid_aucs = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    valid_loss, valid_acc, valid_auc = validate(model, valid_loader, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)
    valid_aucs.append(valid_auc)

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Valid - Loss: {valid_loss:.4f}, Acc: {valid_acc:.2f}%, AUC: {valid_auc:.4f}")
    print()

# Accuracy 및 Loss 그래프 그리기
plt.plot(range(1, num_epochs + 1), train_accs, label='Train')
plt.plot(range(1, num_epochs + 1), valid_accs, label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
plt.plot(range(1, num_epochs + 1), valid_losses, label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# AUC 그래프 그리기
plt.plot(range(1, num_epochs + 1), valid_aucs)
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.show()
