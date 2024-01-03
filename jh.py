import random
import pandas as pd
import numpy as np
import os
import cv2
import torch

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, auc, roc_auc_score, roc_curve

import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = {"data_folder": "D:\\abnormal_detection_dataset\\merge_ubi_fights\\",
        "graphs_folder": "./graph/resnet3d/UBI_FIGHTS/", "epoch": 50, "batch_size": 1, "num_class": 1, "learning_rate": 1e-4,  # 원래 1e-4
        "decay_rate": 0.998, "num_workers": 4, "img_size": (640, 360), "img_depth": 3, "FPS": 50, "SEED": 41} # decay_rate 원래 0.98 "img_size": (320, 240), ubi-fight= (640, 360)

if not os.path.exists(args["graphs_folder"]) : os.mkdir(args["graphs_folder"])

model_save_folder = './trained_model/resnet3d/UBI_FIGHTS/'
if not os.path.exists(model_save_folder) : os.mkdir(model_save_folder)


# randomseed 고정시키기
def seed_everything(seed):
    # tensorflow seed 고정
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # numpy seed 고정
    np.random.seed(seed)

    # pytorch seed 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(args["SEED"])  # seed 고정

videos = os.listdir("D:/abnormal_detection_dataset/merge_ubi_fights/train/fight/") + os.listdir("D:/abnormal_detection_dataset/merge_ubi_fights/train/normal/")
labels = [0 for i in range(len(os.listdir("D:/abnormal_detection_dataset/merge_ubi_fights/train/fight/")))] + [1 for i in range(
    len(os.listdir("D:/abnormal_detection_dataset/merge_ubi_fights/train/normal/")))]

train_x, val_x, train_y, val_y = train_test_split(videos, labels, test_size=0.2, random_state=args['SEED'])


# torch.utils.data.Dataset을 상속받아 직접 커스텀 데이터셋 만들기
class CustomDataset(Dataset):  # Dataset함수 오버라이드
    def __init__(self, videos, labels):  # 데이터의 전처리를 해주는 부분
        self.video_path_list = videos
        self.label_list = labels

    def __getitem__(self, index):  # 데이터셋에서 특정 1개의 샘플을 가져오는 함수
        frames = self.get_video(self.video_path_list[index])  # get_video함수에서 프레임 반환

        if self.label_list is not None:  # 라벨 리스트가 None이 아니면 프레임과 label 반환
            label = self.label_list[index]
            return frames, label
        else:  # 라벨 리스트가 None이면 프레임 반환
            return frames

    def __len__(self):  # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
        return len(self.video_path_list)

    def get_video(self, path):
        frames = []
        if path[:2] == "F_":
            path = "D:/abnormal_detection_dataset/merge_ubi_fights/train/fight/" + path

        else:
            path = "D:/abnormal_detection_dataset/merge_ubi_fights/train/normal/" + path

        cap = cv2.VideoCapture(path)
        for _ in range(args['FPS']):
            _, img = cap.read()
            img = cv2.resize(img, args["img_size"])
            img = img / 255.
            frames.append(img)

        # frames shape = (30, 640, 360, 3)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)  # frames shape [3, 30, 640, 360]으로 변경


train_dataset = CustomDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])

val_dataset = CustomDataset(val_x, val_y)
val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"])

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)

        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=5):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BaseModel(nn.Module):
    def __init__(self, num_classes=5):
        super(BaseModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(3, 8, (3, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 32, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((1, 7, 7)),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


model_depth = 101

from datetime import datetime, timezone, timedelta

# 시간 고유값
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join('results', str(model_depth), train_serial)
# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)


def train(model, optimizer, train_loader, val_loader, scheduler, device, args):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_score = 0
    best_model = None
    best_epoch = 0
    y_pred, y_true = [], []
    train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []
    train_preds, train_trues = [], []
    for epoch in range(1, args["epoch"] + 1):
        model.train()
        running_corrects = 0
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(output) > 0.5
            running_corrects += torch.sum(preds == labels.squeeze())

            y_pred.append(output.detach().cpu().numpy())
            y_true.append(labels.detach().cpu().numpy())

            train_preds += output.argmax(1).detach().cpu().numpy().tolist()
            train_trues += labels.detach().cpu().numpy().tolist()

            train_loss_list.append(loss.item())

        train_acc = accuracy_score(train_preds, train_trues)
        epoch_acc = running_corrects / (len(train_loader) * args["batch_size"])

        _train_loss = np.mean(train_loss_list)

        train_acc_list.append(train_acc)
        train_loss_list.append(_train_loss)

        # _val_loss, _val_score, acc, y_preds, y_trues = validation(model, criterion, val_loader, device)
        # print(f'Epoch {epoch}/{args["EPOCHS"]} Train Loss: {_train_loss:.5f} Train Acc: {epoch_acc:.5f} Val Loss : {_val_loss:.5f} Val Acc: {acc:.5f} Val F1 : {_val_score:.5f}')
        print(f'Epoch {epoch}/{args["epoch"]} Train Loss: {_train_loss:.5f} Train Acc: {train_acc:.5f} epoch_acc{epoch_acc:.5f}')

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for videos, labels in tqdm(iter(val_loader)):
                videos = videos.to(device)
                labels = labels.to(device)

                logit = model(videos)

                loss = criterion(logit, labels)

                test_loss_list.append(loss.item())

                preds += logit.argmax(1).detach().cpu().numpy().tolist()
                trues += labels.detach().cpu().numpy().tolist()

                y_pred.append(logit.detach().cpu().numpy())
                y_true.append(labels.detach().cpu().numpy())

            _val_loss = np.mean(test_loss_list)

        _val_score = f1_score(trues, preds, average='macro')
        val_acc = accuracy_score(trues, preds)

        test_acc_list.append(val_acc)
        test_loss_list.append(_val_loss)

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
            best_epoch = epoch
            print('best model found!')
            torch.save(model.state_dict(), os.path.join(RECORDER_DIR, "best-model.pt"))
        print(f'val_acc: {val_acc:.5f} _val_loss: {_val_loss:.5f} epoch_acc{epoch_acc:.5f}')
        print('best F1 : ', best_val_score, ', best epoch : ', best_epoch)

    print(type(y_pred), type(y_true))
    y_pred = np.array(y_pred, dtype=np.int64)  # .sum(axis=0)
    y_pred = y_pred.astype(np.int64)
    y_true = np.array(y_true, dtype=np.int64)  # .max(axis=1)

    print(type(y_pred), type(y_true))
    print(y_pred.shape, y_true.shape)

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    print(y_pred.shape, y_true.shape)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=np.array(y_pred), average='macro')  # , multi_class='ovr'
    print('AUC:', auc)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')  # , multi_class='ovr'
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc='lower right')
    plt.savefig(args["graphs_folder"] + args["data_folder"].split('\\')[-1] + str(
        args["img_size"]) + f"model_depth{model_depth}" + f"auc_07041339.png")
    plt.show()

    print('AUC: ', roc_auc)

    train_acc_list = torch.Tensor(train_acc_list).detach().cpu().numpy().tolist()
    test_acc_list = torch.Tensor(test_acc_list).detach().cpu().numpy().tolist()
    train_loss_list = torch.Tensor(train_loss_list).detach().cpu().numpy().tolist()
    test_loss_list = torch.Tensor(test_loss_list).detach().cpu().numpy().tolist()


    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.title('Train and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"] + args["data_folder"].split('\\')[-1] + str(
        args["img_size"]) + f"model_depth{model_depth}" + f"acc_07041339.png")
    plt.show()

    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.title('Train and Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"] + args["data_folder"].split('\\')[-1] + str(
        args["img_size"]) + f"model_depth{model_depth}" + f"loss_07041339.png")
    plt.show()
# def validation(model, criterion, val_loader, device):
#     model.eval()
#     val_loss = []
#     preds, trues = [], []
#     y_preds, y_trues = [], []
#     with torch.no_grad():
#         for videos, labels in tqdm(iter(val_loader)):
#             videos = videos.to(device)
#             labels = labels.to(device)
#
#             logit = model(videos)
#
#             loss = criterion(logit, labels)
#
#             val_loss.append(loss.item())
#
#             preds += logit.argmax(1).detach().cpu().numpy().tolist()
#             trues += labels.detach().cpu().numpy().tolist()
#
#             y_preds.append(logit.detach().cpu().numpy())
#             y_trues.append(labels.detach().cpu().numpy())
#
#         _val_loss = np.mean(val_loss)
#
#     _val_score = f1_score(trues, preds, average='macro')
#     acc = accuracy_score(trues, preds)
#
#     return _val_loss, _val_score, acc, y_preds, y_trues


kwargs = {'n_input_channels': 3,
          'conv1_t_size': 7,
          'conv1_t_stride': 1,
          'no_max_pool': False,
          'shortcut_type': 'B',
          'widen_factor': 1.0,
          'n_classes': 2}

if model_depth == 10:
    model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
elif model_depth == 18:
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
elif model_depth == 34:
    model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
elif model_depth == 50:
    model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
elif model_depth == 101:
    model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
elif model_depth == 152:
    model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
elif model_depth == 200:
    model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

model.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=.0004)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,
                                                                 T_mult=2, eta_min=0.00001)
if __name__ == '__main__':
    train(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=.0004), train_loader=train_loader, val_loader=val_loader,
          scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001), device=device, args=args)
# infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)