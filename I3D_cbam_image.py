import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import label_binarize
# from torchvision import transforms

from sklearn.metrics import f1_score, accuracy_score, auc, roc_auc_score, roc_curve

import cv2, os, glob
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

# import GPUtil
# GPUtil.showUtilization()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

args = {"data_folder": "D:\\abnormal_detection_dataset\\RWF_2000_Dataset\\",
        "graphs_folder": "./graph_I3D/UBI-FIGHTS-IMAGE/", "epoch": 50, "batch_size": 1, "num_class": 1,
        "learning_rate": 0.0001,  # 원래 1e-4
        "decay_rate": 0.998, "num_workers": 4, "img_size": (224, 224), "img_depth": 3,
        "FPS": 110}  # decay_rate 원래 0.98 "img_size": (320, 240)
model_save_folder = './trained_model_I3D/UBI-FIGHTS-IMAGE/'

device = torch.device('cuda:0')


## 2d
############################################################### 2d cbam ###################################################
# class Channel_Attention(nn.Module):
#
#     def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
#         #  channel_in: 입력 피쳐 맵의 채널 수를 나타내는 매개변수로 전달
#         #  reduction_ratio: 채널 어텐션에서 사용되는 감소 비율(reduction ratio)을 나타내는 매개변수로, 기본값은 16
#         super(Channel_Attention, self).__init__()
#         self.pool_types = pool_types
#
#         self.shared_mlp = nn.Sequential(  # self.shared_mlp은 채널 어텐션에서 사용되는 공유 다층 퍼셉트론(MLP)을 나타내는 nn.Sequential 모듈
#             nn.Flatten(),
#             nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),  # 입력 피쳐 맵의 채널 수를 감소 비율에 따라 줄이는 fully-connected 레이어
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in)   # 감소된 채널 수를 원래의 채널 수로 확장하는 fully-connected 레이어
#         )
#
#
#     def forward(self, x):
#
#         channel_attentions = []
#
#         for pool_types in self.pool_types:
#             if pool_types == 'avg':
#                 # pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 2d
#                 pool_init = nn.AvgPool3d(kernel_size=(1, x.size(2), x.size(3)), stride=(1, x.size(2), x.size(3)))  # 3d
#                 avg_pool = pool_init(x)
#                 channel_attentions.append(self.shared_mlp(avg_pool))  # x에 대해 avgpool2d 연산 후 avg_pool 변수에 저장하고 그걸 __init__의 self.shared_mlp 통과시킨 뒤 channel_attentions 리스트에 추가
#             elif pool_types == 'max':
#                 # pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 위와 같고 maxpooling2d 연산을 한다는 것만 다름  2d
#                 pool_init = nn.AvgPool3d(kernel_size=(1, x.size(2), x.size(3)), stride=(1, x.size(2), x.size(3)))  # 3d
#                 max_pool = pool_init(x)
#                 channel_attentions.append(self.shared_mlp(max_pool))
#
#         pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)  # 각 풀링에 대한 channel attention 결과를 합쳐서(가장 앞 차원 의미하는 dim=0 기준으로 결과 합침)pooling_sums 변수에 저장
#         # 그 후 .sum(dim=0)을 통해 첫 번째 차원을 따라 합산을 수행하여 각 위치에 대해 channel_attentions 리스트의 원소들을 합친 결과를 얻는다.
#         # 이렇게 얻어진 pooling_sums 텐서에는 avg와 max 풀링의 결과에 대한 channel attention 값들이 합산되어 저장되어 있다., pooling_sums 텐서의 크기는 (batch_size, channel)
#         # scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)  # nn.Sigmoid()를 통해 합쳐진 채널 어텐션 결과에 시그모이드 함수를 적용하여 0~1 값으로 변환 > channel attention의 스케일링을 나타냄.  2d
#         # unsqueeze(2).unsqueeze(3)를 통해 pooling_sums 텐서의 크기는 (batch_size, channel, 1, 1)이 됨. 최종 결과: 입력 데이터 x와 동일한 크기로 확장
#         # .expand_as(x) 통해 pooling_sums 텐서의 크기를 x 텐서와 동일하게 확장함. x 텐서는 입력으로 주어진 텐서로, pooling_sums 텐서와 같은 크기를 가지게 된다.
#         # 이렇게 얻어진 scaled 텐서에는 pooling_sums 텐서의 값을 x 텐서와 동일한 크기로 확장한 결과가 저장되어 있다.
#         # scaled 텐서의 크기는 x 텐서((batch_size, channel_in, height, width))와 동일한 크기를 가지게 된다.
#         ############ scaled 텐서의 크기는 (batch_size, channel, height, width)
#         # unsqueeze(2)와 unsqueeze(3)를 통해 pooling_sums 텐서에 2개의 차원을 추가하고, expand_as(x)를 통해 x 텐서와 동일한 크기로 확장
#         scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).unsqueeze(4)
#         scaled = scaled.expand_as(x)
#
#         return x * scaled  # return the element-wise multiplication between the input and the result.
#         # x와 scaled를 element-wise 곱한 결과를 반환하는 부분이고, x는 원본 입력 텐서이고, scaled는 채널 어텐션을 적용한 결과이다
#         # 두 텐서의 크기가 동일하므로 element-wise 곱셈이 가능하다. 이 연산을 통해 입력 텐서 x에 채널 어텐션을 적용한 결과를 얻을 수 있다.
#
# class ChannelPool(nn.Module):
#     # 피쳐맵의 모든 채널들을 두 개의 개별 채널로 병합. 첫 번째 채널은 모든 채널에서 !최댓값!을 가져와 생성하고 두 번째 채널은 모든 채널에서 !평균!을 가져와 생성한다.
#     def forward(self, x):  # 입력 피쳐 맵의 모든 채널을 최대값과 평균값을 사용하여 두 개의 채널로 병합하는 기능
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
#     # torch.max(x, 1)[0].unsqueeze(1): 입력 텐서 x의 모든 채널에 대해서 최대값을 계산하고, 최대값들을 채널 차원에 대해 제거한 결과를 반환한 것을 2d 텐서로 만듦
#     # torch.mean(x, 1).unsqueeze(1): 입력 텐서 x의 모든 채널에 대해서 평균값을 계산하고, 평균값들을 채널 차원에 대해 제거한 결과를 반환한 것을 2d 텐서로 만듦
#     # torch.cat, dim=1: 위에서 얻은 두 개의 텐서를 채널 차원을 기준으로 병합한다. 이를 통해 최대값과 평균값을 각각 따로 가진 두 개의 채널을 병합한 결과를 얻을 수 있다.
#
# # class Spatial_Attention(nn.Module):  # 2d
# #
# #     def __init__(self, kernel_size=7): # kernel_size를 인자로 받음
# #
# #         # Spatial_Attention 아키텍처
# #         super(Spatial_Attention, self).__init__()
# #
# #         self.compress = ChannelPool() # 위의 ChannelPool 객체 생성
# #         self.spatial_attention = nn.Sequential(  # 객체 생성
# #             nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
# #             nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
# #         )
# #
# #
# #     def forward(self, x):
# #         x_compress = self.compress(x)  # ChannelPool 객체인 self.compress(x) 호출해 입력 텐서에 대해 연산
# #         x_output = self.spatial_attention(x_compress)  # 위의 연산 결과를 spatial_attention을 호출해 Spatial Attention 적용한 결과 x_output에 저장
# #         scaled = nn.Sigmoid()(x_output)  # sigmoid 통해 값을 0~1로 만듦
# #         return x * scaled  # x와 활성화 값인 scaled를 곱하여 element-wise 곱셈을 수행한 후, 그 결과를 반환: 입력 텐서 x를 활성화 값 scaled로 가중치를 적용하여 출력을 생성하는 동작임.
# class Spatial_Attention(nn.Module): ## 3d
#
#     def __init__(self, kernel_size=7):
#         super(Spatial_Attention, self).__init__()
#
#         self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(1, kernel_size, kernel_size), stride=(1, 1, 1),
#                               padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
#         self.bn = nn.BatchNorm3d(1)
#         self.act_fn = nn.ReLU(inplace=True)
#
#
#     def forward(self, x):
#         # 입력 텐서 크기를 (batch_size, channels, depth, height, width)로 변경
#         x = x.unsqueeze(2)
#         x_compress = torch.cat((torch.max(x, 2)[0].unsqueeze(2), torch.mean(x, 2).unsqueeze(2)), dim=2)
#
#         # Conv2d 대신 Conv3d를 사용하여 합성곱 연산 수행
#         x_out = self.conv(x_compress)
#         x_out = self.bn(x_out)
#         x_out = self.act_fn(x_out)
#
#         # 5차원 텐서를 다시 4차원으로 변경
#         x_out = x_out.squeeze(2)
#
#         return x_out  # 3d
#
#
# class CBAM(nn.Module):
#     # CBAM 아키텍처
#     # def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True): # CBAM 클래스의 생성자 함수, 인자로 channel_in(입력 채널 수), reduction_ratio(채널 축소 비율), 풀링 타입, Spatial channel의 활성화 여부
#     def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']): # CBAM 클래스의 생성자 함수, 인자로 channel_in(입력 채널 수), reduction_ratio(채널 축소 비율), 풀링 타입, Spatial channel의 활성화 여부
#         super(CBAM, self).__init__()  # 상속
#         # self.spatial = spatial # Spatial channel의 활성화 여부를 나타냄. boolean 값
#
#         self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types) # 채널 어텐션 모듈 객체를 생성
#
#         # if self.spatial: # spatial=True 이면
#         self.spatial_attention = Spatial_Attention(kernel_size=7) # Spatial_Attention 생성
#
#
#     def forward(self, x):
#         x_out = self.channel_attention(x) # 입력 데이터 x에 대해 채널 어텐션 적용 후 x_out에 저장
#         # if self.spatial: # spatial=True 이면
#         x_out = self.spatial_attention(x_out) # 위의 x_out에 Spatial_Attention 적용
#
#         return x_out
############################################################### 2d cbam ###################################################

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


class Conv3d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(Conv3d_BN, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# I3D 모델에 CBAM을 적용한 인셉션 모듈 >> 논문 같이 전체 I3D가 아님
class InceptionModule_with_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule_with_CBAM, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels[0], kernel_size=1),
            nn.BatchNorm3d(out_channels[0]),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(out_channels[0], out_channels[1], kernel_size=1),
            nn.BatchNorm3d(out_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels[1], out_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels[2]),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(out_channels[2], out_channels[3], kernel_size=1),
            nn.BatchNorm3d(out_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels[3], out_channels[4], kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels[4]),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(out_channels[4], out_channels[5], kernel_size=1),
            nn.BatchNorm3d(out_channels[5]),
            nn.ReLU(inplace=True),
        )
        nc = out_channels[0] + out_channels[2] + out_channels[4] + out_channels[5]
        self.cbam = CBAM(nc, reduction_ratio=16)
        # self.cbam = CBAM(in_channels, reduction_ratio=16)
        # print('in_channels:: ', in_channels)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        x4 = self.branch4(x3)

        # 인셉션 모듈의 출력을 결합
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.cbam(x)
        return x


# I3D 모델에 CBAM을 적용한 인셉션 모듈 >> 논문과 같은 구조
class I3D_with_CBAM(nn.Module):
    def __init__(self, num_classes=args["num_class"]):
        super(I3D_with_CBAM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        ###### Given groups=1, weight of size [23, 368!!!!!!!!!!!!!!!!, 1, 1, 1], expected input[1, 256, 1, 1, 1] to have 368 channels, but got 256 channels instead
        self.inception1 = InceptionModule_with_CBAM(64, [64, 96, 128, 16, 32,  32])  # 64 + 96 + 128 + 16 + 32 + 32 = 368, 256
        ##### Given groups=1, weight of size [23, 368, 1, 1, 1], expected input[1, 256, 1, 1, 1] to have 368 channels, but got 256!!!!!!!!!!!!!! channels instead
        self.inception2 = InceptionModule_with_CBAM(256, [128, 128, 192, 32, 96, 64])  # 640, 480
        self.inception3a = InceptionModule_with_CBAM(480, [192, 96, 208, 16, 48, 64])  # 624, 512
        self.inception3b = InceptionModule_with_CBAM(512, [160, 112, 224, 24, 64, 64])  # 648, 512
        self.inception4a = InceptionModule_with_CBAM(512, [128, 128, 256, 24, 64, 64])  # 664, 512
        self.inception4b = InceptionModule_with_CBAM(512, [112, 144, 288, 32, 64, 64])  # 704, 512
        self.inception5a = InceptionModule_with_CBAM(528, [256, 160, 320, 32, 128, 128])  # 1024, 528
        self.inception5b = InceptionModule_with_CBAM(832, [384, 192, 384, 48, 128, 128])  # 1264, 832

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Sequential(nn.Linear(in_features=64, out_features=num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        # print(x.shape)
        x = self.avg_pool(x)
        x = torch.nn.Flatten()(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)

        return x

gc.collect()
torch.cuda.empty_cache()

#### videocapture 이용한 customdataset #####
class CustomDataset(Dataset):

    def __init__(self, dataset_folder_path=args["data_folder"], image_size=args["img_size"], image_depth=3, train=True, transform=None):
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
        # fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        for frame in range(args["FPS"]):
            ret, image = cap.read()

            if not ret:
                continue
            elif self.image_depth == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, args["img_size"])
                # print('color image size: ', image.shape)  # (240, 320, 3)
                # print(width, height)
                # print(self.read_folder(self.train))
                image = image / 255
                frames.append(image)
                # cv2.imshow('img', image)
                # print(image.shape)
                # cv2.waitKey()
                # print(np.array(frames).shape)
                # frames.append(image)
                # print('dnl', image.shape)
                # print(label)
                # cv2.imshow('3', image)
                # cv2.waitKey()
            else:
                image = cv2.resize(image, args["img_size"])
                image = image / 255
                frames.append(image)
            # print(np.array(frames).shape)
            # print(torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2).shape)  # torch.Size([3, 32, 240, 320])
            # cv2.imshow('img', image)
            # cv2.waitKey()
            # print(np.array(frames).shape)
            # print(image.shape)

            # print(image.shape)
            # print(label)
            # cv2.imshow('0', image)
            # cv2.waitKey()
        # if self.image_depth == 1:
        #     image = cv2.imread(image, 0)
        # else:
        #     image = cv2.imread(image)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if self.transform:
        #     image = self.transform(image)
        #         frames.append(image)
        # frames = np.array(frames)
        # print(frames)
        # print('=============================')
        # a = torch.tensor(frames)
        # print(a.shape)

        # return image, label
        # return {'image': image, 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장
        # return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2), {'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장
        # a = torch.tensor(frames)
        # print(a)
        # print(frames)
        # a = np.array(frames)
        # print(a.shape)pr
        # print('frames', np.array(frames).shape, label)
        # return {'image': torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2), 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장

        # return {'image': torch.FloatTensor(image).permute(2, 0, 1), 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장

        return {'image': torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2), 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장
        # return {'image': torch.FloatTensor(np.array(frames)), 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장

    def get_classnames(self):
        # print('---------------------------')
        # print(os.listdir(f"{self.dataset_folder_path}/train/"))   # ['Fight', 'NonFight'],  UBI-FIGHTS: ['fight', 'normal']
        return os.listdir(f"{self.dataset_folder_path}train/")  # ['Fight', 'NonFight']

    def read_folder(self):
        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path}train/"
            # print('train folder_path: ', folder_path)
        else:
            folder_path = f"{self.dataset_folder_path}val/"  # valid도 가능
            # print('val folder_path: ', folder_path)

        for x in glob.glob(folder_path + "**", recursive=True):  # 해당 폴더의 하위 폴더까지 탐색
            if not x.endswith('avi'):  # rwf2000: avi, ubi-fights: mp4
                continue
            # print("x.split('\\')", x.split('\\'))
            class_idx = self.classes.index(x.split('\\')[-2])  # 클래스 이름(Violence, Nonviolence)의 인덱스 저장 # ubi, ucfcrime
            # print('class_idx: ', class_idx, ' ', 'classes.index(x.split("\\")[-2]: ', x.split('\\')[-2])
            # print(x, x.split('\\')[-2], class_idx) # D:\abnormal_detection_dataset\RWF_2000_Dataset\train\Fight\fi98.avi Fight 0
            ############################## Fight가 0, NonFight가 1 ############################## : RWF2000
            ############################## fight가 0, normal이 1 ############################## : UBI-FIGHTS
            image_path_label.append(
                (x, int(class_idx)))  # ('D:\\abnormal_detection_dataset\\RWF_2000_Dataset\\train\\Fight\\fi243.avi', 0)
            # for i in image_path_label:
            #     if 1 in image_path_label:
            #         print(i)
            #     else:
            #         print('fu')
            # print('image_path_label: ', image_path_label) # 'D:\\abnormal_detection_dataset\\RWF_2000_Dataset\\train\\Fight\\fi243.avi', 0) # Fight:0, NonFight: 1 UBI-FIGHTS도 동일
        return image_path_label  # (이미지 경로, 클래스 인덱스)


#### 콜백함수

# class CustomDataset(Dataset):
#
#     def __init__(self, dataset_folder_path=args["data_folder"], image_size=64, image_depth=3, train=True, transform=None):
#         self.dataset_folder_path = dataset_folder_path
#         self.transform = transform  # HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경
#         self.image_size = image_size
#         self.image_depth = image_depth
#         self.train = train
#         self.classes = sorted(self.get_classnames())
#         self.image_path_label = self.read_folder()
#
#     def __len__(self):
#         # print(self.image_path_label)
#         return len(self.image_path_label)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         image, label = self.image_path_label[idx]
#
#         if self.image_depth == 1:
#             image = cv2.imread(image, 0)
#         else:
#             image = cv2.imread(image)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, self.image_size)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # cv2.imshow('image', image)
#         # cv2.waitKey()
#
#         if self.transform:
#             image = self.transform(image)
#
#         # return image, label
#         return {'image': image, 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장
#
#     # def __getitem__(self, idx):
#     #     frames = self.get_video(self.)
#
#     def get_classnames(self):
#         # print('---------------------------')
#         # print(os.listdir(f"{self.dataset_folder_path.rstrip('/')}/Train/"))
#         return os.listdir(f"{self.dataset_folder_path}Train/")  # violence, nonviolence
#
#     def read_folder(self):
#         image_path_label = []
#
#         if self.train:
#             folder_path = f"{self.dataset_folder_path}train/"
#         else:
#             folder_path = f"{self.dataset_folder_path}val/"  # valid도 가능
#
#         for x in glob.glob(folder_path + "**", recursive=True):  # 해당 폴더의 하위 폴더까지 탐색
#             if not x.endswith('png'):
#                 continue
#
#             class_idx = self.classes.index(x.split('\\')[-2])  # 클래스 이름(Violence, Nonviolence)의 인덱스 저장 # ubi, ucfcrime
#             # print("self.classes.index(x.split('\\')[-2]):", x.split('\\')[-2])
#             # print(class_idx, x.split('\\')[-2])
#             image_path_label.append(
#                 (x, int(class_idx)))  # (D:/InChang/SCVD/frames_new_split_train/train\NonViolence\nv59\nv59-0223.jpg, 0)
#             # print('image_path_label: ', image_path_label)  # fight:0, nonfight:1
#         return image_path_label  # (이미지 경로, 클래스 인덱스)

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, verbose=False, delta=0,
                 path=f"{model_save_folder}I3D_cbam_UBI-FIGHTS-IMAGE_640x360_model_early.pth"):
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


##### 가중치 초기화 함수
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# def custom_collate(batch):
#     inputs = [item['image'] for item in batch]
#     labels = [item['label'] for item in batch]
#
#     # 텐서 크기를 변경하기 전에 복제(clone)를 사용
#     inputs = torch.stack(inputs, dim=0).clone()
#     labels = torch.Tensor(labels).clone()
#
#     return {'image': inputs, 'label': labels}
torch.cuda.empty_cache()


def train(args, device: "cuda:0"):
    cnt = 0
    model = I3D_with_CBAM(num_classes=args["num_class"])
    initialize_weights(model)

    model.to(device)
    # summary(model, (3, 360, 240))

    # 옵티마이저 정의
    optimizer = optim.SGD(model.parameters(), lr=args["learning_rate"], momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    # optimizer = Adam(model.parameters(), args["learning_rate"])
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=args["learning_rate"])
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

    # lr_decay = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args["decay_rate"])
    lr_decay = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    # 손실 함수 정의
    criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)

    ##### train, test dataset 정의
    train_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"],
                                  image_depth=args["img_depth"],
                                  train=True,
                                  transform=transforms.Compose(
                                      [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))  # transforms.ToTensor(): HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경  , transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(180, expand=False)
    # transforms=transforms.ToTensor())  # transforms.ToTensor(): HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경
    test_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"],
                                 image_depth=args["img_depth"],
                                 train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    # 학습 데이터 로더 정의
    train_generator = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,
                                 num_workers=args["num_workers"],
                                 pin_memory=True)  # , sampler=sampler , collate_fn=custom_collate
    test_generator = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True,
                                num_workers=args["num_workers"],
                                pin_memory=True)  # , collate_fn=custom_collate

    early_stopping = EarlyStopping(verbose=True)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    best_accuracy = 0
    valid_losses = []
    y_true = []
    y_pred = []

    # 학습
    for epoch in range(args["epoch"]):
        model.train()
        loss_per_epoch = []
        accuracy_per_epoch = []
        gc.collect()
        torch.cuda.empty_cache()
        batch_accuracy = 0

        for i, train_data in tqdm(enumerate(train_generator)):
            torch.cuda.empty_cache()
            # inputs, labels = train_data['image'], train_data['label']  # torch.Size([16, 3, 150, 240, 320]) # video i3d+cbam
            # inputs = torch.FloatTensor(np.array(inputs)).permute(2, 1, 0, 3, 4)  # torch.Size([150, 3, 16, 240, 320]) # video i3d+cbam
            # inputs, labels = torch.FloatTensor(np.array(inputs)).permute(2, 1, 0, 3, 4).to(device, non_blocking=True), labels.to( device, non_blocking=True)      # # video i3d+cbam

            inputs, labels = train_data['image'], train_data['label']  # torch.Size([16, 3, 150, 240, 320]) # image i3d+cbam
            # print('inputs.shape, labels.shape: ', inputs.shape, labels.shape)
            # inputs = torch.FloatTensor(np.array(inputs)).permute(0, 2, 1, 3, 4)  # torch.Size([150, 3, 16, 240, 320]) # image i3d+cbam
            inputs, labels = torch.FloatTensor(np.array(inputs)).to(device, non_blocking=True), labels.to(device, non_blocking=True)  # # image i3d+cbam
            # print(type(inputs), type(labels))
            labels = labels.unsqueeze(1)

            # print('처음', inputs.shape, labels.shape)
            # 처음 torch.Size([4, 3, 2200, 240, 360]) torch.Size([4])

            # print(labels.shape)
            # print('둘', inputs.shape, labels.shape)
            # 둘 torch.Size([2200, 3, 4, 240, 360]) torch.Size([4])

            # print(inputs.shape)
            # inputs, labels = train_data['image'].to(device, non_blocking=True), train_data['label'].to(device, non_blocking=True)
            # labels = labels.unsqueeze(1)
            # print('막', inputs.shape, labels.shape)
            # 막 torch.Size([4, 3, 2200, 240, 360]) torch.Size([4, 1])

            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            # print(inputs.shape, labels.shape, outputs.shape)
            # train_loss = criterion(input=outputs.to(torch.float32), target=labels.to(torch.float32))
            train_loss = criterion(input=outputs.to(torch.float32), target=labels.to(torch.float32))

            # 역전파
            train_loss.backward()
            optimizer.step()

            num_data = labels.size()[0]
            preds = torch.argmax(outputs, dim=1)
            correct_pred = torch.sum(preds == labels)
            # batch_accuracy = correct_pred * (100/num_data)  # accuracy 이상하게 나옴
            # batch_accuracy = correct_pred * (100/len(labels))
            batch_accuracy = correct_pred * (100 / num_data)

            loss_per_epoch.append(train_loss.item())
            accuracy_per_epoch.append(batch_accuracy.item())

            # threshold = 0.5                                     # Train accuracy: , 1.00000
            # predicted = (outputs > threshold).float()           # Train accuracy: , 1.00000
            # accuracy = (predicted == labels).float().mean()     # Train accuracy: , 1.00000

            # threshold = 0.5
            # predicted = (outputs > threshold).float()
            # correct = (predicted == labels).float().sum()
            # accuracy = correct / len(labels)

        curr_train_accuracy = sum(accuracy_per_epoch) / (i + 1)
        curr_train_loss = sum(loss_per_epoch) / (i + 1)

        train_loss_list.append(curr_train_loss)
        train_acc_list.append(curr_train_accuracy)

        print(f"Epoch {epoch + 1}/{args['epoch']}")
        # print('=' * 100)
        # print(f"Train Loss : {curr_train_loss:.5f}, Train accuracy : {(batch_accuracy_ / total):.5f}")
        # print('=' * 100)
        # print(f"Training Loss : {curr_train_loss}, Training accuracy : {batch_accuracy.item():.5f}")
        # print(f"Training Loss : {curr_train_loss}, Training accuracy : {batch_accuracy:.5f}")
        # print(f"Training Loss : {curr_train_loss}, Training accuracy : {curr_train_accuracy.item():.5f}")
        print(f"Training Loss : {curr_train_loss:.5f}, Training accuracy : {curr_train_accuracy:.5f}")
        # print(f"Train_loss :  {curr_train_loss:.5f}, 'Train accuracy: , {accuracy:.5f}")
        del curr_train_accuracy
        del curr_train_loss
        del num_data

        model.eval()
        loss_per_epoch = []
        accuracy_per_epoch = []
        batch_accuracy = 0

        with torch.no_grad():
            for i, test_data in tqdm(enumerate(test_generator)):
                # inputs, labels = test_data['image'], test_data['label']                 # video i3d+cbam
                # inputs = torch.FloatTensor(np.array(inputs)).permute(2, 1, 0, 3, 4)     # video i3d+cbam
                # inputs, labels = torch.FloatTensor(np.array(inputs)).permute(2, 1, 0, 3, 4).to(device, non_blocking=True), labels.to( device, non_blocking=True)    # video i3d+cbam

                # inputs, labels = test_data['image'].to(device, non_blocking=True), test_data['label'].to(device, non_blocking=True)
                # inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                inputs, labels = test_data['image'], test_data['label']  # torch.Size([16, 3, 150, 240, 320]) # video i3d+cbam
                # inputs = torch.FloatTensor(np.array(inputs)).permute(0, 2, 1, 3, 4)  # torch.Size([150, 3, 16, 240, 320]) # video i3d+cbam
                # inputs, labels = torch.FloatTensor(np.array(inputs)).permute(0, 2, 1, 3, 4).to(device, non_blocking=True), labels.to(device, non_blocking=True)  # # video i3d+cbam
                inputs, labels = torch.FloatTensor(np.array(inputs)).to(device, non_blocking=True), labels.to(device, non_blocking=True)  # # image i3d+cbam
                labels = labels.unsqueeze(1)

                outputs = model(inputs)
                valid_loss = criterion(input=outputs.to(torch.float32), target=labels.to(torch.float32))
                # valid_loss = criterion(input=outputs, target=labels)

                num_data = labels.size()[0]
                preds = torch.argmax(outputs, dim=1)
                correct_pred = torch.sum(preds == labels)
                batch_accuracy = correct_pred * (100 / num_data)  # accuracy 이상하게 나옴
                # batch_accuracy = correct_pred * (100 / len(labels))

                loss_per_epoch.append(valid_loss.item())
                accuracy_per_epoch.append(batch_accuracy)

                # threshold = 0.5                                   # Valid accuracy: , 1.00000
                # predicted = (outputs > threshold).float()         # Valid accuracy: , 1.00000
                # accuracy = (predicted == labels).float().mean()   # Valid accuracy: , 1.00000

                # threshold = 0.5
                # predicted = (outputs > threshold).float()
                # correct = (predicted == labels).float().sum()
                # accuracy = correct / len(labels)

                # 이진 분류 AUC 구하기
                y_pred.append(outputs.detach().cpu().numpy())
                y_true.append(labels.detach().cpu().numpy())

            curr_test_accuracy = sum(accuracy_per_epoch) / (i + 1)
            curr_test_loss = sum(loss_per_epoch) / (i + 1)

            test_loss_list.append(curr_test_loss)
            test_acc_list.append(curr_test_accuracy)

            valid_losses.append(curr_test_loss)

        # print(f"Valid loss :  {curr_test_loss:.5f}, 'Valid accuracy: , {batch_accuracy.item():.5f}")
        # print(f"Valid loss :  {curr_test_loss:.5f}, 'Valid accuracy: , {batch_accuracy:.5f}")
        # print(f"Valid loss :  {curr_test_loss:.5f}, 'Valid accuracy: , {accuracy:.5f}")
        # print(f"Valid loss :  {curr_test_loss:.5f}, 'Valid accuracy: , {curr_test_accuracy.item():.5f}")
        print(f"Valid loss :  {curr_test_loss:.5f}, 'Valid accuracy: , {curr_test_accuracy:.5f}")
        cnt += 1
        # del curr_test_accuracy
        # del curr_test_loss
        # del num_data

        if epoch % 5 == 0:
            lr_decay.step()
            curr_lr = 0
            for params in optimizer.param_groups:
                curr_lr = params['lr']
            print(f"The current learning rate for training is : {curr_lr}")

        if best_accuracy < curr_test_accuracy:
            torch.save(model.state_dict(), f"{model_save_folder}I3D_cbam_UBI-FIGHTS_640x360_image_model.pth")
            best_accuracy = curr_test_accuracy
            print('Current model has best valid accuracy')

        # print('\n--------------------------------------------------------------------------------\n')
        valid_loss = np.average(valid_losses)
        valid_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # print(type(train_acc_list))
        # train_acc_list = np.array(train_acc_list).detach().cpu().numpy().tolist()
        # test_acc_list = np.array(test_acc_list).detach().cpu().numpy().tolist()
        # train_loss_list = np.array(train_loss_list).detach().cpu().numpy().tolist()
        # test_loss_list = np.array(test_loss_list).detach().cpu().numpy().tolist()
        # print('train_acc_list: ', '\n', train_acc_list)
        # print('test_acc_list: ', '\n', test_acc_list)
        # print('train_loss_list: ', '\n', train_loss_list)
        # print('test_loss_list: ', '\n', test_loss_list)
    train_acc_list = torch.Tensor(train_acc_list).detach().cpu().numpy().tolist()
    test_acc_list = torch.Tensor(test_acc_list).detach().cpu().numpy().tolist()
    train_loss_list = torch.Tensor(train_loss_list).detach().cpu().numpy().tolist()
    test_loss_list = torch.Tensor(test_loss_list).detach().cpu().numpy().tolist()

    print('train_acc_list: ', train_acc_list)
    print('test_acc_list: ', test_acc_list)
    print('train_loss_list: ', train_loss_list)
    print('test_loss_list: ', test_loss_list)
    # print(type(train_acc_list))
    # plt.plot(train_acc_list)
    # plt.plot(test_acc_list)
    # plt.title('Train and Validation Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('epoch')
    # plt.xlim([1, epoch])
    # plt.legend(['train', 'valid'], loc='upper left')
    # plt.savefig(args["graphs_folder"]+f"UBI-FIGHTS_video_224x224_acc_graph.png")
    # # plt.show()
    #
    # plt.plot(train_loss_list)
    # plt.plot(test_loss_list)
    # plt.title('Train and Validation Loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.xlim([1, epoch])
    # plt.legend(['train', 'valid'], loc='upper left')
    # plt.savefig(args["graphs_folder"]+f"UBI-FIGHTS_video_224x224_loss_graph.png")
    # plt.show()

    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.title('Train and Validation Accuracy after train')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.xlim([1, int(cnt)])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"] + f"RWF_image_640x360_acc_graph_after_train.png")
    plt.show()

    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.title('Train and Validation Loss after train')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([1, int(cnt)])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"] + f"RWF_image_640x360_loss_graph_after_train.png")
    plt.show()

    # gpt auc 시작
    # print(type(y_pred), type(y_true))
    # y_pred = np.array(y_pred, dtype=np.int64)  # .sum(axis=0)
    # y_pred = y_pred.astype(np.int64)
    # y_true = np.array(y_true, dtype=np.int64)
    # print(type(y_pred), type(y_true))
    # print(y_pred.shape, y_true.shape)
    # # y_pred = y_pred.reshape(-1)
    # # y_true = y_true.reshape(-1)
    # print(y_pred.shape, y_true.shape)
    #
    # ### ucf crime auc ###
    # print(type(y_pred), type(y_true))              # <class 'list'> <class 'list'>
    # y_pred = np.array(y_pred, dtype=np.int64) #.sum(axis=0)
    # y_pred = y_pred.astype(np.int64)
    # y_true = np.array(y_true, dtype=np.int64) #.max(axis=1)
    # print(y_pred.shape, y_true.shape)               # (9, 32, 14) (9, 32)
    # print(type(y_pred), type(y_true))               # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    # print(y_pred.sum(axis=0).shape, y_true.shape)   # (32, 14) (9, 32)
    # print(y_pred.shape, y_true.shape)               # (4032,) (288,) reshape(-1)
    # print(y_pred.shape, y_true.shape)              # (4032,) (288,)  reshape(-1)
    # print(y_pred.max(axis=2).shape, y_true.shape)   # axis=1 >> (9, 14) (9, 32), axis=0 >> (32, 14) (9, 32), axis=2 >> (9, 32)
    # y_pred = y_pred.max(axis=2)
    # print(y_pred.shape, y_true.shape)
    # print('y_pred', '\n', len(y_pred))
    # print('-' * 100)
    # print('y_true', '\n', len(y_true))
    # y_pred = y_pred.reshape(-1)
    # y_true = y_true.reshape(-1)
    # labels = [0, 1]
    # y_true = label_binarize(y_true, classes=labels)
    # y_pred = label_binarize(y_pred, classes=labels)
    # print(y_pred.shape, y_true.shape)               # (4032, 14) (288, 14)
    # print(y_pred.max(axis=1).shape, y_true.shape)
    from sklearn.metrics import roc_curve, auc

    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(args["num_class"]):
    #     fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    #     roc_auc[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
    #     print(roc_auc[i])
    #
    # # Plot of a ROC curve for a specific class
    # plt.figure(figsize=(15, 5))
    # for idx, i in enumerate(range(args["num_class"])):
    #     # plt.subplot(131 + idx)
    #     plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    #     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    #     plt.plot([0, 1], [0, 1], 'k--')
    #     plt.plot([0, 1], [1, 1], 'y--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Class %0.0f' % idx)
    #     plt.legend(loc="lower right")
    #     plt.savefig(args["graphs_folder"]+f"RWF_AUC of Class" + str(idx) + ".png")
    # plt.show()
    #
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    # roc_auc["micro"] = roc_auc_score(y_true.ravel(), y_pred.ravel(), multi_class='ovr', average='macro')
    #
    # plt.plot([0, 1], [0, 1], 'k--',)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Multi-Class ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    ### ucf crime auc ###

    # y_pred = np.concatenate(y_pred, axis=0)
    # y_true = np.concatenate(y_true, axis=0)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=np.array(y_pred), average='macro')
    print('AUC:', auc)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
    plt.title('Receiver Operating Characteristic!')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc='lower right')
    plt.savefig(args["graphs_folder"] + f"RWF_image_640x360_AUC_graph.png")
    plt.show()

    print('AUC: ', roc_auc)
    # gpt auc 끝

    # 내가 쓰던 auc
    print(type(y_pred), type(y_true))  # <class 'list'> <class 'list'>
    y_pred = np.array(y_pred, dtype=np.int64)  # .sum(axis=0)
    y_pred = y_pred.astype(np.int64)
    y_true = np.array(y_true, dtype=np.int64)  # .max(axis=1)
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
    plt.savefig(args["graphs_folder"] + f"RWF_imageAUC_graph!.png")
    # plt.show()

    print('AUC: ', roc_auc)
    # 내가 쓰던 auc 끝

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=np.array(y_pred), average='macro')
    print('AUC:', auc)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
    plt.title('Receiver Operating Characteristic after train')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc='lower right')
    plt.savefig(args["graphs_folder"] + f"RWF_image_640x360_AUC_graph_after_train.png")
    plt.show()

    print('train_loss_list: ', train_loss_list)
    print('train_acc_list: ', train_acc_list)
    print('test_loss_list: ', test_loss_list)
    print('test_acc_list: ', test_acc_list)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == '__main__':
    train(args=args, device=device)