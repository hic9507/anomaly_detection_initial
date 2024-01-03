import random
import pandas as pd
import numpy as np
import os
import cv2
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    "FPS": 500,  # 프레임 갯수
    "IMG_SIZE": 256,  # 비디오 이미지 사이즈
    "EPOCHS": 200,  # 에폭
    "LEARNING_RATE": 3e-4,  # 학습률
    "BATCH_SIZE": 2,  # 배치사이즈
    "SEED": 41  # 넘파이, 토치등등 랜덤 시드
}


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


seed_everything(CFG["SEED"])  # seed 고정

videos = os.listdir("D:/abnormal_detection_dataset/UBI_FIGHTS/videos/videos/train/fight/") + os.listdir("D:/abnormal_detection_dataset/UBI_FIGHTS/videos/videos/train/normal/")
labels = [0 for x in range(len(os.listdir("D:/RWF_2000_Dataset/train/Fight/")))] + [1 for y in range(len(os.listdir("D:/RWF_2000_Dataset/train/NonFight/")))]

train_x, val_x, train_y, val_y = train_test_split(videos, labels, test_size=0.2, random_state=CFG['SEED'])


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
        if path[:2] == "fi":
            path = "D:/abnormal_detection_dataset/UBI_FIGHTS/videos/videos/train/fight/" + path

        else:
            path = "D:/abnormal_detection_dataset/UBI_FIGHTS/videos/videos/train/normal/" + path

        cap = cv2.VideoCapture(path)
        for _ in range(CFG['FPS']):
            _, img = cap.read()
            img = cv2.resize(img, (CFG["IMG_SIZE"], CFG["IMG_SIZE"]))
            img = img / 255.
            frames.append(img)

        # frames shape = (30, 640, 360, 3)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)  # frames shape [3, 30, 640, 360]으로 변경


train_dataset = CustomDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_x, val_y)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# def get_inplanes():
#     return [64, 128, 256, 512]
#
#
# def conv3x3x3(in_planes, out_planes, stride=1):
#     return nn.Conv3d(in_planes,
#                      out_planes,
#                      kernel_size=3,
#                      stride=stride,
#                      padding=1,
#                      bias=False)
#
#
# def conv1x1x1(in_planes, out_planes, stride=1):
#     return nn.Conv3d(in_planes,
#                      out_planes,
#                      kernel_size=1,
#                      stride=stride,
#                      bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super().__init__()
#
#         self.conv1 = conv3x3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#
#         out = self.relu(out)
#
#         out = self.conv2(out)
#
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super().__init__()
#
#         self.conv1 = conv1x1x1(in_planes, planes)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = conv3x3x3(planes, planes, stride)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = conv1x1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm3d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#
#         out = self.relu(out)
#
#         out = self.conv2(out)
#
#         out = self.bn2(out)
#
#         out = self.relu(out)
#
#         out = self.conv3(out)
#
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
#                  no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=5):
#         super().__init__()
#
#         block_inplanes = [int(x * widen_factor) for x in block_inplanes]
#
#         self.in_planes = block_inplanes[0]
#         self.no_max_pool = no_max_pool
#
#         self.conv1 = nn.Conv3d(n_input_channels,
#                                self.in_planes,
#                                kernel_size=(conv1_t_size, 7, 7),
#                                stride=(conv1_t_stride, 2, 2),
#                                padding=(conv1_t_size // 2, 3, 3),
#                                bias=False)
#         self.bn1 = nn.BatchNorm3d(self.in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
#                                        shortcut_type)
#         self.layer2 = self._make_layer(block,
#                                        block_inplanes[1],
#                                        layers[1],
#                                        shortcut_type,
#                                        stride=2)
#         self.layer3 = self._make_layer(block,
#                                        block_inplanes[2],
#                                        layers[2],
#                                        shortcut_type,
#                                        stride=2)
#         self.layer4 = self._make_layer(block,
#                                        block_inplanes[3],
#                                        layers[3],
#                                        shortcut_type,
#                                        stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight,
#                                         mode='fan_out',
#                                         nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _downsample_basic_block(self, x, planes, stride):
#         out = F.avg_pool3d(x, kernel_size=1, stride=stride)
#         zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
#                                 out.size(3), out.size(4))
#         if isinstance(out.data, torch.cuda.FloatTensor):
#             zero_pads = zero_pads.cuda()
#
#         out = torch.cat([out.data, zero_pads], dim=1)
#
#         return out
#
#     def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
#         downsample = None
#         if stride != 1 or self.in_planes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(self._downsample_basic_block,
#                                      planes=planes * block.expansion,
#                                      stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     conv1x1x1(self.in_planes, planes * block.expansion, stride),
#                     nn.BatchNorm3d(planes * block.expansion))
#
#         layers = []
#         layers.append(
#             block(in_planes=self.in_planes,
#                   planes=planes,
#                   stride=stride,
#                   downsample=downsample))
#         self.in_planes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.in_planes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         if not self.no_max_pool:
#             x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
#
# class BaseModel(nn.Module):
#     def __init__(self, num_classes=5):
#         super(BaseModel, self).__init__()
#         self.feature_extract = nn.Sequential(
#             nn.Conv3d(3, 8, (3, 3, 3)),
#             nn.ReLU(),
#             nn.BatchNorm3d(8),
#             nn.MaxPool3d(2),
#             nn.Conv3d(8, 32, (2, 2, 2)),
#             nn.ReLU(),
#             nn.BatchNorm3d(32),
#             nn.MaxPool3d(2),
#             nn.Conv3d(32, 64, (2, 2, 2)),
#             nn.ReLU(),
#             nn.BatchNorm3d(64),
#             nn.MaxPool3d(2),
#             nn.Conv3d(64, 128, (2, 2, 2)),
#             nn.ReLU(),
#             nn.BatchNorm3d(128),
#             nn.MaxPool3d((1, 7, 7)),
#         )
#         self.classifier = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.feature_extract(x)
#         x = x.view(batch_size, -1)
#         x = self.classifier(x)
#         return x


# model_depth = 101
class Channel_Attention(nn.Module):

    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        #  channel_in: 입력 피쳐 맵의 채널 수를 나타내는 매개변수로 전달
        #  reduction_ratio: 채널 어텐션에서 사용되는 감소 비율(reduction ratio)을 나타내는 매개변수로, 기본값은 16
        super(Channel_Attention, self).__init__()
        self.pool_types = pool_types

        self.shared_mlp = nn.Sequential(  # self.shared_mlp은 채널 어텐션에서 사용되는 공유 다층 퍼셉트론(MLP)을 나타내는 nn.Sequential 모듈
            nn.Flatten(),
            nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),  # 입력 피쳐 맵의 채널 수를 감소 비율에 따라 줄이는 fully-connected 레이어
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in)# 감소된 채널 수를 원래의 채널 수로 확장하는 fully-connected 레이어
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))  # x에 대해 avgpool2d 연산 후 avg_pool 변수에 저장하고 그걸 __init__의 self.shared_mlp 통과시킨 뒤 channel_attentions 리스트에 추가
                # channel_attentions.append(self.shared_mlp(avg_pool.view(avg_pool.size(0), -1))) # GPT
                channel_attention = self.shared_mlp(avg_pool.view(avg_pool.size(0), -1))
                channel_attentions.append(channel_attention)

            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 위와 같고 maxpooling2d 연산을 한다는 것만 다름
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool)) # 원래 코드
                # channel_attentions.append(self.shared_mlp(max_pool.view(max_pool.size(0), -1))) # GPT

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)  # 각 풀링에 대한 channel attention 결과를 합쳐서(가장 앞 차원 의미하는 dim=0 기준으로 결과 합침)pooling_sums 변수에 저장
        # 그 후 .sum(dim=0)을 통해 첫 번째 차원을 따라 합산을 수행하여 각 위치에 대해 channel_attentions 리스트의 원소들을 합친 결과를 얻는다.
        # 이렇게 얻어진 pooling_sums 텐서에는 avg와 max 풀링의 결과에 대한 channel attention 값들이 합산되어 저장되어 있다., pooling_sums 텐서의 크기는 (batch_size, channel)
        scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)  # nn.Sigmoid()를 통해 합쳐진 채널 어텐션 결과에 시그모이드 함수를 적용하여 0~1 값으로 변환 > channel attention의 스케일링을 나타냄.
        # unsqueeze(2).unsqueeze(3)를 통해 pooling_sums 텐서의 크기는 (batch_size, channel, 1, 1)이 됨. 최종 결과: 입력 데이터 x와 동일한 크기로 확장
        # .expand_as(x) 통해 pooling_sums 텐서의 크기를 x 텐서와 동일하게 확장함. x 텐서는 입력으로 주어진 텐서로, pooling_sums 텐서와 같은 크기를 가지게 된다.
        # 이렇게 얻어진 scaled 텐서에는 pooling_sums 텐서의 값을 x 텐서와 동일한 크기로 확장한 결과가 저장되어 있다.
        # scaled 텐서의 크기는 x 텐서((batch_size, channel_in, height, width))와 동일한 크기를 가지게 된다.
        ############ scaled 텐서의 크기는 (batch_size, channel, height, width)
        # unsqueeze(2)와 unsqueeze(3)를 통해 pooling_sums 텐서에 2개의 차원을 추가하고, expand_as(x)를 통해 x 텐서와 동일한 크기로 확장

        return x * scaled  # return the element-wise multiplication between the input and the result. # 원래 코드
        # x와 scaled를 element-wise 곱한 결과를 반환하는 부분이고, x는 원본 입력 텐서이고, scaled는 채널 어텐션을 적용한 결과이다
        # 두 텐서의 크기가 동일하므로 element-wise 곱셈이 가능하다. 이 연산을 통해 입력 텐서 x에 채널 어텐션을 적용한 결과를 얻을 수 있다.

class ChannelPool(nn.Module):
    # 피쳐맵의 모든 채널들을 두 개의 개별 채널로 병합. 첫 번째 채널은 모든 채널에서 !최댓값!을 가져와 생성하고 두 번째 채널은 모든 채널에서 !평균!을 가져와 생성한다.
    def forward(self, x):  # 입력 피쳐 맵의 모든 채널을 최대값과 평균값을 사용하여 두 개의 채널로 병합하는 기능
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
    # torch.max(x, 1)[0].unsqueeze(1): 입력 텐서 x의 모든 채널에 대해서 최대값을 계산하고, 최대값들을 채널 차원에 대해 제거한 결과를 반환한 것을 2d 텐서로 만듦
    # torch.mean(x, 1).unsqueeze(1): 입력 텐서 x의 모든 채널에 대해서 평균값을 계산하고, 평균값들을 채널 차원에 대해 제거한 결과를 반환한 것을 2d 텐서로 만듦
    # torch.cat, dim=1: 위에서 얻은 두 개의 텐서를 채널 차원을 기준으로 병합한다. 이를 통해 최대값과 평균값을 각각 따로 가진 두 개의 채널을 병합한 결과를 얻을 수 있다.

class Spatial_Attention(nn.Module):

    def __init__(self, kernel_size=7): # kernel_size를 인자로 받음

        # Spatial_Attention 아키텍처
        super(Spatial_Attention, self).__init__()

        self.compress = ChannelPool() # 위의 ChannelPool 객체 생성
        self.spatial_attention = nn.Sequential(  # 객체 생성
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )


    def forward(self, x):
        x_compress = self.compress(x)  # ChannelPool 객체인 self.compress(x) 호출해 입력 텐서에 대해 연산
        x_output = self.spatial_attention(x_compress)  # 위의 연산 결과를 spatial_attention을 호출해 Spatial Attention 적용한 결과 x_output에 저장
        scaled = nn.Sigmoid()(x_output)  # sigmoid 통해 값을 0~1로 만듦
        return x * scaled  # x와 활성화 값인 scaled를 곱하여 element-wise 곱셈을 수행한 후, 그 결과를 반환: 입력 텐서 x를 활성화 값 scaled로 가중치를 적용하여 출력을 생성하는 동작임.

class CBAM(nn.Module):
    # CBAM 아키텍처
    # def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True): # CBAM 클래스의 생성자 함수, 인자로 channel_in(입력 채널 수), reduction_ratio(채널 축소 비율), 풀링 타입, Spatial channel의 활성화 여부
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']): # CBAM 클래스의 생성자 함수, 인자로 channel_in(입력 채널 수), reduction_ratio(채널 축소 비율), 풀링 타입, Spatial channel의 활성화 여부
        super(CBAM, self).__init__()  # 상속
        # self.spatial = spatial # Spatial channel의 활성화 여부를 나타냄. boolean 값

        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types) # 채널 어텐션 모듈 객체를 생성

        # if self.spatial: # spatial=True 이면
        self.spatial_attention = Spatial_Attention(kernel_size=7) # Spatial_Attention 생성


    def forward(self, x):
        x_out = self.channel_attention(x) # 입력 데이터 x에 대해 채널 어텐션 적용 후 x_out에 저장
        # if self.spatial: # spatial=True 이면
        x_out = self.spatial_attention(x_out) # 위의 x_out에 Spatial_Attention 적용

        return x_out

##### Densenet 정의
class DenseBottleNeck(nn.Module):
# class DenseBlock(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseBottleNeck, self).__init__()

        inner_channel = growth_rate * bn_size

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, inner_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channel)
        self.conv2 = nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.drop_rate > 0:
            out = self.dropout(out, p=self.drop_rate, training=self.training)
        out = torch.cat([x, out], 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print('x.shape: ', x.shape)
        out = self.bn1(x)
        # print('out.shape1: ', out.shape)
        out = self.relu(out)
        # print('out.shape2: ', out.shape)
        out = self.conv1(out)
        # print('out.shape3: ', out.shape)
        out = self.bn2(out)
        # print('out.shape4: ', out.shape)
        out = self.relu(out)
        # print('out.shape5: ', out.shape)
        out = self.conv2(out)
        # print('out.shape6: ', out.shape)
        out = self.dropout(out)
        # print('out.shape7: ', out.shape)
        # if self.drop_rate > 0:
        #     out = self.dropout(out) # , p=self.drop_rate, training=self.training
        out = torch.cat([x, out], 1)
        # print('out.shape8: ', out.shape)
        return out

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0.2, num_classes=2):
    # def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=3, bn_size=4, drop_rate=0.2, num_classes=args["num_class"]):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.denseblock1 = self._make_dense_block(num_layers=block_config[0], num_input_features=num_init_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.cbam1 = CBAM(channel_in=num_init_features + block_config[0] * growth_rate)

        self.transition1 = self._make_transition(num_input_features=num_init_features + block_config[0] * growth_rate, num_output_features=num_init_features + block_config[1] * growth_rate)
        # self.transition1 = self._make_transition(num_input_features=num_init_features + block_config[0] * growth_rate, num_output_features=num_init_features)  # 수정
        self.denseblock2 = self._make_dense_block(num_layers=block_config[1], num_input_features=num_init_features + block_config[1] * growth_rate, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        # self.cbam2 = CBAM(channel_in=num_init_features + block_config[1] * growth_rate)

        self.cbam2 = CBAM(channel_in=num_init_features + block_config[0] * growth_rate + num_init_features + block_config[1] * growth_rate + 128)



    # self.cbam2 = CBAM(channel_in=num_init_features + block_config[1] * growth_rate + num_init_features + block_config[2] * growth_rate)  # 수정

        self.transition2 = self._make_transition(num_input_features=num_init_features + block_config[1] * growth_rate + 384, num_output_features=num_init_features + block_config[2] * growth_rate)
        # self.transition2 = self._make_transition(num_input_features=num_init_features + block_config[1] * growth_rate, num_output_features=num_init_features)  # 수정
        self.denseblock3 = self._make_dense_block(num_layers=block_config[2], num_input_features=num_init_features + block_config[2] * growth_rate, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        # self.cbam3 = CBAM(channel_in=num_init_features + block_config[2] * growth_rate)

        self.cbam3 = CBAM(channel_in=num_init_features + block_config[2] * growth_rate + num_init_features + block_config[3] * growth_rate + 192)  # 수정

        self.transition3 = self._make_transition(num_input_features=num_init_features + block_config[2] * growth_rate + 768, num_output_features=num_init_features + block_config[3] * growth_rate)
        # self.transition3 = self._make_transition(num_input_features=num_init_features + block_config[2] * growth_rate, num_output_features=num_init_features)  # 수정
        self.denseblock4 = self._make_dense_block(num_layers=block_config[3], num_input_features=num_init_features + block_config[3] * growth_rate, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.cbam4 = CBAM(channel_in=num_init_features + block_config[3] * growth_rate + 512)

        self.bn = nn.BatchNorm2d(num_init_features + block_config[3] * growth_rate + 512)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_init_features + block_config[3] * growth_rate + 512, num_classes)
        # self.fc = nn.Linear(1280, num_classes)

    def _make_dense_block(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        layers = []
        for i in range(num_layers):
            layers.append(DenseBlock(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)) # 6*32+64 = 256
        return nn.Sequential(*layers)

    def _make_transition(self, num_input_features, num_output_features):
        return Transition(num_input_features, num_output_features)

    def forward(self, x):
        # print('0', x.shape)
        x = self.features(x)
        # print('1111111111111', x.shape)
        x = self.denseblock1(x)
        # print('222', x.shape) # 222 torch.Size([2, 256, 90, 160]), 6*32+64 = 256
        x = self.cbam1(x)
        # print('33', x.shape) # cbam 거쳐도 형태 변화없음
        x = self.transition1(x)
        # print('44', x.shape)

        x = self.denseblock2(x)
        # print('5', x.shape)
        x = self.cbam2(x)
        # print('6', x.shape)
        x = self.transition2(x)
        # print('7', x.shape)

        x = self.denseblock3(x)
        # print('8', x.shape)
        x = self.cbam3(x)
        # print('9', x.shape)
        x = self.transition3(x)
        # print('10', x.shape)

        x = self.denseblock4(x)
        # print('11', x.shape)
        x = self.cbam4(x)
        # print('12', x.shape)
        x = self.bn(x)
        # print('13', x.shape)
        x = self.relu(x)
        # print('14', x.shape)
        x = self.avgpool(x)
        # print('15', x.shape)
        x = x.view(x.size(0), -1)
        # print('234', x.shape)
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x

class Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()

        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
from datetime import datetime, timezone, timedelta

# 시간 고유값
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join('results', train_serial)
# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_score = 0
    best_model = None
    best_epoch = 0
    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        val_loss, val_score, acc = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 : [{val_score:.5f}]  Val Acc :[{acc:.5f}]')

        if scheduler is not None:
            scheduler.step(val_score)

        if best_val_score < val_score:
            best_val_score = val_score
            best_model = model
            best_epoch = epoch
            print('best model found!')
            torch.save(model.state_dict(), os.path.join(RECORDER_DIR, "best-model.pt"))

    print('best F1 : ', best_val_score, ', best epoch : ', best_epoch)
    return best_model


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []

    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            logit = model(videos)

            loss = criterion(logit, labels)

            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()

        _val_loss = np.mean(val_loss)

    val_score = f1_score(trues, preds, average='macro')
    acc = accuracy_score(trues, preds)

    return val_loss, val_score, acc


kwargs = {'n_input_channels': 3,
          'conv1_t_size': 7,
          'conv1_t_stride': 1,
          'no_max_pool': False,
          'shortcut_type': 'B',
          'widen_factor': 1.0,
          'n_classes': 2}

# if model_depth == 10:
#     model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
# elif model_depth == 18:
#     model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
# elif model_depth == 34:
#     model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
# elif model_depth == 50:
#     model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
# elif model_depth == 101:
#     model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
# elif model_depth == 152:
#     model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
# elif model_depth == 200:
#     model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

model = DenseNet()

model.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=.0004)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,
                                                                 T_mult=2, eta_min=0.00001)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)