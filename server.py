import random
import pandas as pd
import numpy as np
import os
import cv2
import glob
from tqdm.auto import tqdm
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from functools import partial

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = {"data_folder": "C:\\InChang\\UBI_FIGHTS\\videos\\",
        "graphs_folder": "./graph_resnet3d/UBI-FIGHTS-VIDEO/", "epoch": 10, "batch_size": 1, "num_class": 1,
        "learning_rate": 0.001,  # 원래 1e-4
        "decay_rate": 0.998, "num_workers": 4, "img_size": (640, 360), "img_depth": 3, "FPS": 50,
        "SEED": 41}  # decay_rate 원래 0.98 "img_size": (320, 240)

if not os.path.exists(args["graphs_folder"]): os.mkdir(args["graphs_folder"])


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
                frames.append(image)
            else:
                image = cv2.resize(image, self.image_size)
                image = image / 255
                frames.append(image)

        # return {'image': torch.FloatTensor(image).permute(2, 0, 1), 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)  # frames shape [3, 30, 640, 360]으로 변경

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
            if not x.endswith('mp4'):  # rwf2000: mp4
                continue
            class_idx = self.classes.index(x.split('\\')[-2])  # 클래스 이름(Violence, Nonviolence)의 인덱스 저장 # ubi, ucfcrime
            image_path_label.append(
                (x, int(class_idx)))  # ('D:\\abnormal_detection_dataset\\RWF_2000_Dataset\\train\\Fight\\fi243.avi', 0)
        return image_path_label  # (이미지 경로, 클래스 인덱스)


# train_dataset = CustomDataset(train_x, train_y)
# train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
#
# val_dataset = CustomDataset(val_x, val_y)
# val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

##### train, test dataset 정의
train_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"],
                              image_depth=args["img_depth"],
                              train=True,
                              )  # transforms.ToTensor(): HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경
test_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"],
                             image_depth=args["img_depth"],
                             train=False)
# sampler = RandomSampler(train_dataset)

# train_generator = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"], pin_memory=True)  # , sampler=sampler
train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"],
                          pin_memory=True)  # , sampler=sampler
val_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"],
                        pin_memory=True)


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x_out = self.channel_attention(x)
        x_out = self.spatial_attention(x_out)
        x = x * x_out  # .unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        # print('fc1.shape: ', in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        # print('avg_pool1: ', avg_pool.shape)

        avg_pool = self.fc1(avg_pool)
        # print('avg_pool2: ', avg_pool.shape)

        avg_pool = self.relu(avg_pool)
        # print('avg_pool3: ', avg_pool.shape)

        channel_att = self.fc2(avg_pool)
        # print('channel_att1: ', channel_att.shape)

        channel_att = self.sigmoid(channel_att)
        # print('channel_att2: ', channel_att.shape)

        return x * channel_att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        min_pool = torch.min(x, dim=1, keepdim=True)[0]
        concat = torch.cat([max_pool, min_pool], dim=1)
        spatial_att = self.conv(concat)
        spatial_att = self.sigmoid(spatial_att)

        return x * spatial_att


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

    def __init__(self, in_planes, planes, stride=1, downsample=None, expansion=4):
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
        self.cbam = CBAM(channels=planes * expansion)

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

        out = self.cbam(out)

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
    def __init__(self, num_classes=1):
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

# model_depth = 101
model_depth = 50

# 시간 고유값
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join('results', str(model_depth), train_serial)
# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)


def train(model, optimizer, train_loader, val_loader, scheduler, args, device="cuda:0"):
    model.to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    best_val_score = 0
    best_epoch = 0

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    loss_per_epoch = []
    accuracy_per_epoch = []
    y_true = []
    y_pred = []

    for epoch in range(1, args['epoch'] + 1):
        model.train()

        running_corrects = 0

        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

            optimizer.zero_grad()

            output = model(videos)
            loss = criterion(output, labels)

            preds = torch.sigmoid(output) > 0.5
            running_corrects += torch.sum(preds == labels.squeeze())

            loss_per_epoch.append(loss.item())
            accuracy_per_epoch.append(batch_accuracy.item())

            loss.backward()
            optimizer.step()

            y_pred.append(output.detach().cpu().numpy())
            y_true.append(labels.detach().cpu().numpy())

        epoch_acc = running_corrects / (len(train_loader) * args["batch_size"])

        if len(train_loss_list) == 0:
            train_loss_avg = loss.item()
        else:
            train_loss_avg = sum(train_loss_list) / len(train_loss_list)

        if len(train_acc_list) == 0:
            train_acc_avg = epoch_acc.item()
        else:
            train_acc_avg = sum(train_acc_list) / len(train_acc_list)

        train_loss_list.append(train_loss_avg)
        train_acc_list.append(train_acc_avg)

        print(f"Epoch {epoch}/{args['epoch']}")
        print(f'Train Acc : {train_acc_avg:.5f}  Train Loss : {train_loss_avg:.5f}')  # 최적

        model.eval()
        loss_per_epoch = []
        accuracy_per_epoch = []
        running_corrects = 0
        batch_accuracy = 0

        with torch.no_grad():
            preds, trues = [], []

            for videos, labels in tqdm(iter(val_loader)):

                videos = videos.to(device, non_blocking=True)
                labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

                logit = model(videos)

                test_loss = criterion(logit, labels)

                loss_per_epoch.append(test_loss.item())
                accuracy_per_epoch.append(batch_accuracy.item())

                valid_preds = torch.sigmoid(logit) > 0.5
                running_corrects += torch.sum(valid_preds == labels.squeeze())

                val_epoch_acc_ = running_corrects / (len(val_loader) * args["batch_size"])

                if len(test_loss_list) == 0:
                    val_loss_avg = loss.item()
                else:
                    val_loss_avg = sum(test_loss_list) / len(test_loss_list)

                if len(test_acc_list) == 0:
                    val_acc_avg = val_epoch_acc_
                else:
                    val_acc_avg = sum(test_acc_list) / len(test_acc_list)

                test_loss_list.append(val_loss_avg)
                test_acc_list.append(val_acc_avg)

                preds += logit.argmax(1).detach().cpu().numpy().tolist()
                trues += labels.detach().cpu().numpy().tolist()

                y_pred.append(logit.detach().cpu().numpy())
                y_true.append(labels.detach().cpu().numpy())

        val_score = f1_score(trues, preds, average='macro')
        acc = accuracy_score(trues, preds)
        print(f"  Validation Acc :{val_acc_avg:.5f}  Validation Loss : {val_loss_avg:.5f}")

        if scheduler is not None:
            scheduler.step(val_score)

        if best_val_score < val_score:
            best_val_score = val_score
            best_epoch = epoch
            print('best model found!')
            torch.save(model.state_dict(), os.path.join(RECORDER_DIR, "best-model.pth"))

        print('best F1 : ', best_val_score, ', best epoch : ', best_epoch)

    if epoch == args['epoch']:
        y_pred = np.array(y_pred, dtype=np.int64)  # .sum(axis=0)
        y_pred = y_pred.astype(np.int64)
        y_true = np.array(y_true, dtype=np.int64)

        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

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
        plt.savefig(args["graphs_folder"] + f"resnet3d_cbam_auc.png")
        plt.show()

        print('AUC: ', roc_auc)

        plt.plot(train_acc_list)
        plt.plot(test_acc_list)
        plt.title('Train and Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.xlim([1, args["epoch"]])
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(args["graphs_folder"] + f"resnet3d_cbam_acc.png")
        plt.show()

        plt.plot(train_loss_list)
        plt.plot(test_loss_list)
        plt.title('Train and Validation Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.xlim([1, args["epoch"]])
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(args["graphs_folder"] + f"resnet3d_cbam_loss.png")
        plt.show()


kwargs = {'n_input_channels': 3,
          'conv1_t_size': 7,
          'conv1_t_stride': 1,
          'no_max_pool': False,
          'shortcut_type': 'B',
          'widen_factor': 1.0,
          'n_classes': 1}

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

optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=.0004)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,
                                                                 T_mult=2, eta_min=0.00001)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    device = "cuda:0"
    train(args=args, device=device, model=model,
          optimizer=torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=.0004),
          train_loader=train_loader, val_loader=val_loader,
          scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001))
