import random
import numpy as np
from functools import partial
from datetime import datetime, timezone, timedelta
import os
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import torchvision.transforms as transforms


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = {"data_folder": "D:\\abnormal_detection_dataset\\merge_ubi_fights\\",
        "graphs_folder": "./graph/resnet3d/UBI_FIGHTS/", "epoch": 500, "batch_size": 1, "num_class": 1, "learning_rate": 3e-4,  # 원래 1e-4
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

# RWF
# videos = os.listdir(args['data_folder'] + "train/Fight/") + os.listdir(args['data_folder']+ "train/NonFight/")
# labels = [0 for i in range(len(os.listdir(args['data_folder'] + "train/Fight/")))] + [1 for i in range(len(os.listdir(args['data_folder'] + "train/NonFight/")))]

# UBI
videos = os.listdir(args['data_folder'] + "train/fight/") + os.listdir(args['data_folder']+ "train/normal/")
labels = [0 for i in range(len(os.listdir(args['data_folder'] + "train/fight/")))] + [1 for i in range(len(os.listdir(args['data_folder'] + "train/normal/")))]

train_x, val_x, train_y, val_y = train_test_split(videos, labels, test_size=0.2, random_state=args['SEED'])

model_depth = 50

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=10, verbose=False, delta=0, path=f"{model_save_folder}" + args["data_folder"].split('\\')[-1] + str(args["img_size"]) + f"model_depth{model_depth}" + f"resnet3d_cbam_early.pth"):
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
            print(f'Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  val loss is decreased, Saving model ...')
            print('\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x_out = self.channel_attention(x)
        x_out = self.spatial_attention(x_out)
        x = x * x_out#.unsqueeze(2).unsqueeze(3).unsqueeze(4)
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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.dropout(out)

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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.cbam(out)

        out += residual
        out = self.relu(out)
        out = self.dropout(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=5):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.dropout = nn.Dropout(0.2)
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
                    nn.BatchNorm3d(planes * block.expansion),
                    nn.Dropout(0.2))

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
        x = self.dropout(x)
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
            nn.Dropout(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 32, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Dropout(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Dropout(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Dropout(0.2),
            nn.MaxPool3d((1, 7, 7)),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

# torch.utils.data.Dataset을 상속받아 직접 커스텀 데이터셋 만들기

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# 시간 고유값
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join('results', str(model_depth), train_serial)
# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)

class CustomDataset(Dataset):  # Dataset함수 오버라이드
    def __init__(self, videos, labels, transform=None):  # 데이터의 전처리를 해주는 부분
        self.video_path_list = videos
        self.label_list = labels
        self.transform = transform

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
            path = args['data_folder']+ "train/fight/" + path # ubi fights
            # path = args['data_folder']+ "train/Fight/" + path   # rwf

        else:
            path = args['data_folder'] + "train/normal/" + path     # ubi fights
            # path = args['data_folder'] + "train/NonFight/" + path     # rwf

        cap = cv2.VideoCapture(path)
        for _ in range(args['FPS']):
            _, img = cap.read()
            img = cv2.resize(img, args["img_size"])
            img = img / 255.
            frames.append(img)

        # frames shape = (30, 640, 360, 3)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)  # frames shape [3, 30, 640, 360]으로 변경

train_dataset = CustomDataset(train_x, train_y)     # , transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(180, expand=False)])
train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args["num_workers"])

val_dataset = CustomDataset(val_x, val_y)           # , transform=transforms.Compose([transforms.ToTensor()])
val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args["num_workers"])

def train(model, device, args, train_loader, val_loader):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=0.0001) # .0004
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=0.0001) # .0004
    lr_decay = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    # lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args["decay_rate"])
    # lr_decay = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    initialize_weights(model)

    best_val_score = 0
    best_model = None
    best_epoch = 0
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
    best_accuracy = 0
    valid_losses = []
    y_pred, y_true = [], []
    train_acc_for_graph, test_acc_for_graph, train_loss_for_graph, test_loss_for_graph = [], [], [], []

    early_stopping = EarlyStopping(verbose=True)

    for epoch in range(1, args['epoch'] + 1):

        model.train()

        running_corrects = 0

        for videos, labels in tqdm(iter(train_loader)):

            videos = videos.to(device)
            labels = labels.unsqueeze(1).float().to(device)

            optimizer.zero_grad()

            output = model(videos)
            train_loss = criterion(input=output, target=labels)

            train_loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(output)) > 0.5
            running_corrects += torch.sum(preds == labels.squeeze())

            # num_data = labels.size()[0]
            # # train_pred = torch.argmax(output, dim=1) # 원래 코드
            # # correct_train_pred = torch.sum(train_pred == labels)
            # # batch_accuracy = correct_train_pred * (100 / num_data)

            y_pred.append(output.detach().cpu().numpy())
            y_true.append(labels.detach().cpu().numpy())

        epoch_acc = running_corrects / (len(train_loader) * args["batch_size"])

        train_loss_list.append(train_loss.item())
        train_acc_list.append(epoch_acc.item())            # .item()

        if len(train_loss_list) == 0:
            train_loss_avg = train_loss.item()
        else:
            train_loss_avg = sum(train_loss_list) / len(train_loss_list)

        if len(train_acc_list) == 0:
            train_acc_avg = epoch_acc.item()
        else:
            train_acc_avg = sum(train_acc_list) / len(train_acc_list)

        print(f"Epoch {epoch}/{args['epoch']}")
        print(f"train_loss.item() :  {train_loss.item():.5f} epoch_acc.item():  {epoch_acc.item():.5f}")
        print(f"train_loss :  {train_loss:.5f} epoch_acc:  {epoch_acc:.5f}")
        # print(f"train_loss_avg.item() :  {train_loss_avg.item():.5f} train_acc_avg.item():  {train_acc_avg.item():.5f}")
        print(f"train_loss_avg :  {train_loss_avg:.5f} train_acc_avg:  {train_acc_avg:.5f}")            # 이걸 리스트에 추가해 그래프 그려야함
        train_acc_for_graph.append(train_acc_avg)
        train_loss_for_graph.append(train_loss_avg)

        model.eval()
        loss_per_epoch = []
        ##### gpt val acc calculate
        correct = 0
        total = 0
        epoch_loss_, num_data__ = 0, 0
        moving_avg_window = 5  # 최근 5개 에폭에 대한 평균을 계산
        ##### gpt val acc calculate
        with torch.no_grad():
            # y_pred, y_true = [], []
            preds, trues = [], []
            running_corrects = 0
            for videos, labels in tqdm(iter(val_loader)):
                videos = videos.to(device)
                labels = labels.unsqueeze(1).float().to(device)

                logit = model(videos)
                test_loss = criterion(input=logit, target=labels)
                loss_per_epoch.append(test_loss.item())                     # 더 나은 accuracy 측정 위함(early stopping)

                # print('logit.shape: ', logit.shape)
                # print('logit: ', '\n', logit)
                # print('test_loss: ', '\n', test_loss)
                valid_preds = (torch.sigmoid(logit) > 0.5).float()
                running_corrects += torch.sum(valid_preds == labels.squeeze())
                # print('len(valid_preds), len(labels), valid_preds.shape, labels.shape: ', len(valid_preds), len(labels), valid_preds.shape, labels.shape)
                # print('valid_preds: ', '\n', valid_preds)
                # print('labels: ', '\n', labels)

                ##### gpt val acc calculate
                probabilities = torch.sigmoid(logit)
                predicted = (probabilities > 0.5).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                ##### gpt val acc calculate

                y_pred.append(logit.detach().cpu().numpy())
                y_true.append(labels.detach().cpu().numpy())

                # preds += logit.argmax(1).detach().cpu().numpy().tolist()            # 원래 코드 - cross entropy loss
                preds += valid_preds.detach().cpu().numpy().tolist()            # acc 이상해서 추가함
                trues += labels.detach().cpu().numpy().tolist()                     # acc 이상해서 추가함

                val_epoch_acc = running_corrects / len(val_loader) * args["batch_size"]
                # print('=' * 100)
                # print('val_epoch_acc: ', val_epoch_acc)

                epoch_loss_ += test_loss.item() * videos.size(0)  # gpt     서버에서 가져옴
                num_data__ += videos.size(0)  # gpt                         서버에서 가져옴

            val_epoch_acc_s = running_corrects / (len(val_loader) * args["batch_size"])
            # print('val_epoch_acc_s: ', val_epoch_acc_s)
            # print('=' * 100)

            test_acc_list.append(val_epoch_acc_s)  # 원래 코드
            test_loss_list.append(test_loss.item())

            val_loss_avg_gpt = epoch_loss_ / num_data__  # gpt
            val_acc_avg_gpt = running_corrects.double() / num_data__  # gpt         0.83500만 반복됨

            # 원래 코드
            # if len(test_loss_list) == 0:
            #     val_loss_avgs = test_loss.item()
            # else:
            #     val_loss_avgs = sum(test_loss_list) / len(test_loss_list)
            #
            # if len(test_acc_list) == 0:
            #     val_acc_avg = val_epoch_acc_s.item()
            # else:
            #     val_acc_avg = sum(test_acc_list) / len(test_acc_list)
            # 원래 코드
            if len(test_loss_list) == 0:
                val_loss_avgs = test_loss.item()
            else:
                val_loss_avgs = sum(test_loss_list) / len(test_loss_list)

            if len(test_acc_list) == 0:
                val_acc_avg = val_epoch_acc_s.item()
            else:
                # 최근 moving_avg_window 에폭에 대한 평균 계산
                val_acc_avg = sum(test_acc_list[-moving_avg_window:]) / len(test_acc_list[-moving_avg_window:])

            # test_acc_list.append(val_acc_avg)                               # 원래 코드
            curr_test_loss = sum(loss_per_epoch) / epoch                    # 더 나은 accuracy 측정 위함(early stopping)

            val_loss_avg = sum(test_loss_list) / len(test_loss_list)  # Calculate average for the epoch
            val_epoch_acc_ = running_corrects / len(val_loader) * args["batch_size"]
            # test_loss_list.append(val_loss_avg)
            # test_acc_list.append(val_acc_avg)
            accuracy = correct / total                                      ##### gpt val acc calculate
            accs = accuracy_score(trues, preds)
        acc = accuracy_score(trues, preds)
        print(f'test_loss.item() : {test_loss.item():.5f}  val_epoch_acc_.item() : {val_epoch_acc_.item():.5f} acc: {acc:.5f} accs: {accs:.5f}')  # 최적
        print(f'test_loss : {test_loss:.5f}  val_epoch_acc_ : {val_epoch_acc_:.5f}  val_epoch_acc: {val_epoch_acc:.5f}  val_epoch_acc.item(): {val_epoch_acc.item():.5f}')  # 최적
        # print(f'val_loss_avg.item() : {val_loss_avg.item():.5f}  val_acc_avg.item() : {val_acc_avg.item():.5f} ')  # 최적
        print(f'val_loss_avg : {val_loss_avg:.5f}  val_acc_avg : {val_acc_avg:.5f} accuracy: {accuracy:.5f} val_epoch_acc_s: {val_epoch_acc_s:.5f}')  # # 이걸 리스트에 추가해 그래프 그려야함
        print(f'val_loss_avgs : {val_loss_avgs:.5f}  val_loss_avg_gpt : {val_loss_avg_gpt:.5f} val_acc_avg_gpt: {val_acc_avg_gpt:.5f} val_epoch_acc_s: {val_epoch_acc_s:.5f}')  # 최적
        test_acc_for_graph.append(val_acc_avg)
        test_loss_for_graph.append(val_loss_avg)
        # valid_losses.append(curr_test_loss)  # 더 나은 accuracy 측정 위함(early stopping)
        valid_losses.append(val_loss_avg)  # 더 나은 accuracy 측정 위함(early stopping)

        if epoch % 5 == 0:
            lr_decay.step()
            curr_lr = 0
            for params in optimizer.param_groups:
                curr_lr = params['lr']
            print(f"The current learning rate for training is : {curr_lr}")

        if best_accuracy < val_acc_avg:
            torch.save(model.state_dict(),  f"{model_save_folder}" + args["data_folder"].split('\\')[-1] + str(args["img_size"]) + f"model_depth{model_depth}" + f"resnet3d_cbam.pth")
            best_accuracy = val_acc_avg
            print('Current model has best valid accuracy')

        valid_loss = np.average(valid_losses)
        valid_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    # if epoch == args['epoch']:
    print(type(y_pred), type(y_true))
    y_pred = np.array(y_pred, dtype=np.int64)
    y_pred = y_pred.astype(np.int64)
    y_true = np.array(y_true, dtype=np.int64)

    print(type(y_pred), type(y_true))
    print(y_pred.shape, y_true.shape)

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    print(y_pred.shape, y_true.shape)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=np.array(y_pred), average='macro')
    print('AUC:', auc)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc='lower right')
    plt.savefig(args["graphs_folder"] + args["data_folder"].split('\\')[-1] + str(args["img_size"]) + f"model_depth{model_depth}" + f"auc.png")
    plt.show()

    print('AUC: ', roc_auc)

    # train_acc_list = torch.Tensor(train_acc_list).detach().cpu().numpy().tolist()
    # test_acc_list = torch.Tensor(test_acc_list).detach().cpu().numpy().tolist()
    # train_loss_list = torch.Tensor(train_loss_list).detach().cpu().numpy().tolist()
    # test_loss_list = torch.Tensor(test_loss_list).detach().cpu().numpy().tolist()

    train_acc_for_graph = torch.Tensor(train_acc_for_graph).detach().cpu().numpy().tolist()
    test_acc_for_graph = torch.Tensor(test_acc_for_graph).detach().cpu().numpy().tolist()
    train_loss_for_graph = torch.Tensor(train_loss_for_graph).detach().cpu().numpy().tolist()
    test_loss_for_graph = torch.Tensor(test_loss_for_graph).detach().cpu().numpy().tolist()

    plt.plot(train_acc_for_graph)
    plt.plot(test_acc_for_graph)
    plt.title('Train and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"] + args["data_folder"].split('\\')[-1] + str(args["img_size"]) + f"model_depth{model_depth}" + f"acc.png")
    plt.show()

    plt.plot(train_loss_for_graph)
    plt.plot(test_loss_for_graph)
    plt.title('Train and Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"] + args["data_folder"].split('\\')[-1] + str(args["img_size"]) + f"model_depth{model_depth}" + f"loss.png")
    plt.show()


kwargs = {'n_input_channels': 3, 'conv1_t_size': 7, 'conv1_t_stride': 1, 'no_max_pool': False,
          'shortcut_type': 'B', 'widen_factor': 1.0, 'n_classes': 1}

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


# optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=0.001) # .0004
#
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
#
# # infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

if __name__ == '__main__':
    train(args=args, device=device, model=model, train_loader=train_loader, val_loader=val_loader)
