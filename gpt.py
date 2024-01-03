# import os
# from tqdm import tqdm
# from collections import OrderedDict
# import cv2
# import numpy as np
# import glob
# import torch
# from torch.utils.data import DataLoader, Dataset
# import matplotlib.pyplot as plt
# import torch.nn as nn
#
# args = {"data_folder": "D:\\abnormal_detection_dataset\\UBI_FIGHTS\\videos\\videos\\",
#         "graphs_folder": "./graph_I3D/UBI-FIGHTS-VIDEO/", "epoch": 50, "batch_size": 1, "num_class": 2, "learning_rate": 0.001,  # 원래 1e-4
#         "decay_rate": 0.998, "num_workers": 4, "img_size": (224, 224), "img_depth": 3, "FPS": 110}
# class CBAM(nn.Module):
#     def __init__(self, channels, reduction_ratio=16):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(channels, reduction_ratio)
#         self.spatial_attention = SpatialAttention()
#
#     def forward(self, x):
#         x_out = self.channel_attention(x)
#         x_out = self.spatial_attention(x_out)
#         x = x * x_out#
#         return x
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_pool = self.avg_pool(x)
#         avg_pool = self.fc1(avg_pool)
#         avg_pool = self.relu(avg_pool)
#         channel_att = self.fc2(avg_pool)
#         channel_att = self.sigmoid(channel_att)
#
#         return x * channel_att
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size % 2 == 1, "Kernel size must be odd."
#         padding = kernel_size // 2
#         self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         min_pool = torch.min(x, dim=1, keepdim=True)[0]
#         concat = torch.cat([max_pool, min_pool], dim=1)
#         spatial_att = self.conv(concat)
#         spatial_att = self.sigmoid(spatial_att)
#
#         return x * spatial_att
#
# class Conv3d_BN(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
#         super(Conv3d_BN, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm3d(out_channels)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x
#
# class InceptionModule_with_CBAM(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(InceptionModule_with_CBAM, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels[0], kernel_size=1),
#             nn.BatchNorm3d(out_channels[0]),
#             nn.ReLU(inplace=True),
#         )
#
#         self.branch2 = nn.Sequential(
#             nn.Conv3d(out_channels[0], out_channels[1], kernel_size=1),
#             nn.BatchNorm3d(out_channels[1]),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels[1], out_channels[2], kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels[2]),
#             nn.ReLU(inplace=True),
#         )
#
#         self.branch3 = nn.Sequential(
#             nn.Conv3d(out_channels[2], out_channels[3], kernel_size=1),
#             nn.BatchNorm3d(out_channels[3]),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels[3], out_channels[4], kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels[4]),
#             nn.ReLU(inplace=True),
#         )
#
#         self.branch4 = nn.Sequential(
#             nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
#             nn.Conv3d(out_channels[4], out_channels[5], kernel_size=1),
#             nn.BatchNorm3d(out_channels[5]),
#             nn.ReLU(inplace=True),
#         )
#         nc= out_channels[0]+out_channels[2]+out_channels[4]+out_channels[5]
#         self.cbam = CBAM(nc, reduction_ratio=16)
#
#     def forward(self, x):
#         x1 = self.branch1(x)
#         x2 = self.branch2(x1)
#         x3 = self.branch3(x2)
#         x4 = self.branch4(x3)
#
#         x = torch.cat((x1, x2, x3, x4), 1)
#         x = self.cbam(x)
#         return x
#
# class I3D_with_CBAM(nn.Module):
#     def __init__(self, num_classes=args["num_class"]):
#         super(I3D_with_CBAM, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)),
#             nn.BatchNorm3d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         )
#         self.inception1 = InceptionModule_with_CBAM(64, [64, 96, 128, 16, 32, 32])
#         self.inception2 = InceptionModule_with_CBAM(256, [128, 128, 192, 32, 96, 64])
#         self.inception3a = InceptionModule_with_CBAM(480, [192, 96, 208, 16, 48, 64])
#         self.inception3b = InceptionModule_with_CBAM(512, [160, 112, 224, 24, 64, 64])
#         self.inception4a = InceptionModule_with_CBAM(512, [128, 128, 256, 24, 64, 64])
#         self.inception4b = InceptionModule_with_CBAM(512, [112, 144, 288, 32, 64, 64])
#         self.inception5a = InceptionModule_with_CBAM(528, [256, 160, 320, 32, 128, 128])
#         self.inception5b = InceptionModule_with_CBAM(832, [384, 192, 384, 48, 128, 128])
#
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.dropout = nn.Dropout(0.3)
#         self.linear1 = nn.Linear(1024, 512)
#         self.linear2 = nn.Linear(512, 256)
#         self.linear3 = nn.Linear(256, 64)
#         self.linear4 = nn.Sequential(nn.Linear(in_features=64, out_features=num_classes), nn.Sigmoid())
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.inception1(x)
#         x = self.inception2(x)
#         x = self.inception3a(x)
#         x = self.inception3b(x)
#         x = self.inception4a(x)
#         x = self.inception4b(x)
#         x = self.inception5a(x)
#         x = self.inception5b(x)
#
#         x= self.avg_pool(x)
#         x=torch.nn.Flatten()(x)
#         x=self.dropout(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)
#         x = self.linear4(x)
#
#         return x
#
# path = '.\\test_I3D_cbam\\UBI_FIGHTS\\'
#
# device = torch.device("cuda:0")
#
#
# class LoadInputImages(Dataset):
#
#     def __init__(self, input_folder=args["data_folder"], image_size=args["img_size"], image_depth=3, train=False,
#                  transform=None):
#         self.input_folder = input_folder
#         self.image_size = image_size
#         self.image_depth = image_depth
#         self.transform = transform
#
#         self.video_paths = self.read_folder()
#
#     def read_folder(self):
#         video_paths = []
#         for x in glob.glob(self.input_folder + '**'):
#             print('x: ', x)
#             if not x.endswith('mp4'):
#                 continue
#             class_idx = x.split('\\')
#             # class_idx_ = class_idx[0]
#             print('class_idx: ', class_idx[-1][0])
#             if class_idx[-1][0] == 'F':
#                 class_idx_ = 1
#             else:
#                 class_idx_ = 0
#             video_paths.append((x, int(class_idx_)))
#
#         return video_paths
#
#     def __len__(self):
#         return len(self.video_paths)
#
#
#
#     def __getitem__(self, idx):
#         frame_list = []
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         image, label = self.video_paths[idx]
#         cap = cv2.VideoCapture(image)
#         for i in range(args["FPS"]):
#             ret, image = cap.read()
#             if not ret:
#                 return None
#             elif self.image_depth == 3:
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = cv2.resize(image, self.image_size)
#                 image = image / 255
#                 frame_list.append(image)
#             else:
#                 image = cv2.resize(image, self.image_size)
#                 image = image / 255
#                 frame_list.append(image)
#         # return torch.FloatTensor(np.array(frame_list)).permute(3, 0, 1, 2)
#         frame_list = np.array(frame_list)
#         frame_list = torch.FloatTensor(frame_list)
#         frame_list = frame_list.permute(0, 3, 1, 2)
#         # 배치 차원과 시간 축 차원 추가
#         # image = frame_list.unsqueeze(0)  # 배치 차원 추가
#         # image = image.unsqueeze(2)  # 시간 축 차원 추가
#         print('image.shape: ', np.array(frame_list).shape)
#
#         # return image, label  # image와 label을 반환
#         return frame_list
#
# # def custom_collate_fn(batch):
# #     return [item for sublist in batch for item in sublist]
#
# def viz(args, device="cuda:0"):
#     model_save_folder = './trained_model_I3D/'
#     model = I3D_with_CBAM(num_classes=args["num_class"])
#
#     model = model.to(device)
#
#     assert os.path.exists(f"{model_save_folder}UBI-FIGHTS-VIDEO/I3D_cbam_UBI-FIGHTS-VIDEO_224x224_model_early.pth"), 'A trained model does not exist!'
#
#     try:
#         state_dict = torch.load(f"{model_save_folder}UBI-FIGHTS-VIDEO/I3D_cbam_UBI-FIGHTS-VIDEO_224x224_model_early.pth", map_location=device)
#         new_state_dict = OrderedDict()
#
#         for k,v in state_dict.items():
#             name = k[7:]
#             new_state_dict[name] = v
#
#         model.load_state_dict(new_state_dict)
#         print("Model loaded!")
#     except Exception as e:
#         print(e)
#
#     model.eval()
#
#     input_data = LoadInputImages(input_folder=path, image_size=args["img_size"], image_depth=args["img_depth"])
#     data_generator = DataLoader(input_data, batch_size=1, shuffle=False, num_workers=1) #  collate_fn=custom_collate_fn
#
#     class_names = ['fight', 'normal']
#
#     output_folder = './output_I3D/UBI_FIGHTS/'
#
#     if not os.path.exists(output_folder) : os.mkdir(output_folder)
#
#
#     fig = plt.figure(figsize=(10, 4))
#
#     for i, image_ in tqdm(enumerate(data_generator)):
#         for image in image_[i]:
#             # print('frame.shape: ', frame.shape)
#             # frame = frame.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
#             # image = cv2.resize(frame, dsize=(224, 224))
#
#             plt.clf() # 그림 지움
#
#             image = torch.tensor(image[i]).to(device)
#             output = model(image[i])
#             cnn_filters = output
#
#             softmaxed_output = torch.nn.Softmax(dim=1)(output)
#             predicted_class = class_names[torch.argmax(softmaxed_output).cpu().numpy()]
#
#             cnn_combined_filter = cv2.resize(torch.max(cnn_filters.squeeze(0), 0)[0].detach().cpu().numpy(), (args["img_size"]))
#             heatmap = np.asarray(cv2.applyColorMap(cv2.normalize(cnn_combined_filter, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
#                                 cv2.COLORMAP_JET), dtype=np.float32)
#
#             # input_img = cv2.resize((image.squeeze(0).detach().cpu().numpy().astype(np.uint8)), dsize=(224,224)) # squeeze(0)을 이용해 차원을 축소하고, permute() 함수를 이용해 차원 순서를 변경해 입력
#             print(i, (image[i].detach().cpu().numpy().astype(np.uint8)).shape)
#             print(i, (image[i].permute(1, 0, 2, 3).detach().cpu().numpy().astype(np.uint8)).shape)
#             print(i, (image[i].permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.uint8)).shape)
#             # input_img = cv2.resize((image[i].squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.uint8)), dsize=(224,224)) # squeeze(0)을 이용해 차원을 축소하고, permute() 함수를 이용해 차원 순서를 변경해 입력
#             input_img = image[i].permute(1, 0, 2, 3).detach().cpu().numpy().astype(np.uint8)
#             input_img = cv2.resize(input_img, dsize=(224, 224))
#
#             heatmap_cnn = cv2.addWeighted(np.asarray(input_img, dtype=np.float32), 0.9, heatmap, 0.0025, 0)  # 이때 cv2.addWeighted 함수를 사용하여 두 이미지를 가중치를 더하여 더한 결과를 얻는다
#
#             fig.add_subplot(131)
#             plt.imshow(input_img)
#             plt.title("Input Image")
#             plt.xticks(())
#             plt.yticks(())
#
#             fig.add_subplot(132)
#             plt.imshow(cnn_combined_filter)
#
#             plt.xticks(())
#             plt.yticks(())
#
#             fig.add_subplot(133)
#             plt.imshow(heatmap_cnn)
#             plt.title("Heat Map")
#             plt.xticks(())
#             plt.yticks(())
#
#             fig.suptitle(f"Network's prediction : {predicted_class.capitalize()}", fontsize=20)
#
#             plt.savefig(f'{output_folder}/{i}.png')
#
# if __name__ == '__main__':
#     viz(args=args, device=device)
#


import os
from tqdm import tqdm
from collections import OrderedDict
import cv2
import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
# from train_video import ResNet50

args = {"data_folder": "D:\\abnormal_detection_dataset\\UBI_FIGHTS\\videos\\videos\\",
        "graphs_folder": "./graph_I3D/UBI-FIGHTS-VIDEO/", "epoch": 50, "batch_size": 1, "num_class": 2, "learning_rate": 0.001,  # 원래 1e-4
        "decay_rate": 0.998, "num_workers": 4, "img_size": (224, 224), "img_depth": 3, "FPS": 110}
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

class Conv3d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(Conv3d_BN, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
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
        nc= out_channels[0]+out_channels[2]+out_channels[4]+out_channels[5]
        self.cbam = CBAM(nc, reduction_ratio=16)
        # self.cbam = CBAM(in_channels, reduction_ratio=16)
        #print('in_channels:: ', in_channels)

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
        self.inception1 = InceptionModule_with_CBAM(64, [64, 96, 128, 16, 32, 32])          # 64 + 96 + 128 + 16 + 32 + 32 = 368, 256
        # self.in_channels = self.inception1.in_channels
        # self.cbam1 = CBAM(sum(self.inception1.out_channels))
        ##### Given groups=1, weight of size [23, 368, 1, 1, 1], expected input[1, 256, 1, 1, 1] to have 368 channels, but got 256!!!!!!!!!!!!!! channels instead
        self.inception2 = InceptionModule_with_CBAM(256, [128, 128, 192, 32, 96, 64])       # 640, 480
        # self.in_channels = self.inception2.in_channels
        # self.cbam2 = CBAM(sum(self.inception2.out_channels))

        self.inception3a = InceptionModule_with_CBAM(480, [192, 96, 208, 16, 48, 64])       # 624, 512
        # self.in_channels = self.inception3a.in_channels
        # self.cbam3 = CBAM(sum(self.inception3a.out_channels))

        self.inception3b = InceptionModule_with_CBAM(512, [160, 112, 224, 24, 64, 64])      # 648, 512
        # self.in_channels = self.inception3b.in_channels
        # self.cbam4 = CBAM(sum(self.inception3b.out_channels))

        self.inception4a = InceptionModule_with_CBAM(512, [128, 128, 256, 24, 64, 64])      # 664, 512
        # self.in_channels = self.inception4a.in_channels
        # self.cbam5 = CBAM(sum(self.inception4a.out_channels))

        self.inception4b = InceptionModule_with_CBAM(512, [112, 144, 288, 32, 64, 64])      # 704, 512
        # self.in_channels = self.inception4b.in_channels
        # self.cbam6 = CBAM(sum(self.inception4b.out_channels))

        self.inception5a = InceptionModule_with_CBAM(528, [256, 160, 320, 32, 128, 128])    # 1024, 528
        # self.in_channels = self.inception5a.in_channels
        # self.cbam7 = CBAM(sum(self.inception5a.out_channels))

        self.inception5b = InceptionModule_with_CBAM(832, [384, 192, 384, 48, 128, 128])    # 1264, 832
        # self.in_channels = self.inception5b.in_channels
        # self.cbam8 = CBAM(sum(self.inception5b.out_channels))

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Sequential(nn.Linear(in_features=64, out_features=num_classes), nn.Sigmoid())

    def forward(self, x):
        print('x.shape: ', x.shape)
        # x = torch.tensor(x).permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        print('I3D_with_CBAM conv1: ', x.shape)  # torch.Size([1, 64, 500, 60, 90])
        x = self.inception1(x)
        #inception1_out_channels = self.inception1.cbam.channel_attention.fc1.out_channels
        #x = self.inception1.cbam.channel_attention(x[:, :inception1_out_channels])
        #print('I3D_with_CBAM inception1: ', x.shape)

        x = self.inception2(x)
        #inception2_out_channels = self.inception2.cbam.channel_attention.fc1.out_channels
        #x = self.inception2.cbam.channel_attention(x[:, :inception2_out_channels])
        #print('I3D_with_CBAM inception2: ', x.shape)

        x = self.inception3a(x)
        #inception3a_out_channels = self.inception3a.cbam.channel_attention.fc1.out_channels
        #x = self.inception3a.cbam.channel_attention(x[:, :inception3a_out_channels])
        #print('I3D_with_CBAM inception3a: ', x.shape)

        x = self.inception3b(x)
        #inception3b_out_channels = self.inception3b.cbam.channel_attention.fc1.out_channels
        #x = self.inception3b.cbam.channel_attention(x[:, :inception3b_out_channels])
        #print('I3D_with_CBAM inception3b: ', x.shape)

        x = self.inception4a(x)
        # inception4a_out_channels = self.inception4a.cbam.channel_attention.fc1.out_channels
        # x = self.inception4a.cbam.channel_attention(x[:, :inception4a_out_channels])
        # print('I3D_with_CBAM inception4a: ', x.shape)

        x = self.inception4b(x)
        # inception4b_out_channels = self.inception4b.cbam.channel_attention.fc1.out_channels
        # x = self.inception4b.cbam.channel_attention(x[:, :inception4b_out_channels])
        # print('I3D_with_CBAM inception4b: ', x.shape)

        x = self.inception5a(x)
        # inception5a_out_channels = self.inception5a.cbam.channel_attention.fc1.out_channels
        # x = self.inception5a.cbam.channel_attention(x[:, :inception5a_out_channels])
        # print('I3D_with_CBAM inception5a: ', x.shape)

        x = self.inception5b(x)
        # inception5b_out_channels = self.inception5b.cbam.channel_attention.fc1.out_channels
        # x = self.inception5b.cbam.channel_attention(x[:, :inception5b_out_channels])
        # print('I3D_with_CBAM inception5b: ', x.shape)

        # print(x.shape)
        x= self.avg_pool(x)
        x=torch.nn.Flatten()(x)
        x=self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)

        return x


path = './test_I3D_cbam/UBI_FIGHTS/'

device = torch.device("cuda:0")

class LoadInputImages(Dataset):

    def __init__(self, input_folder=path, image_size=args["img_size"], image_depth=args["img_depth"], transform=None):

        self.input_folder = input_folder
        self.image_size = image_size
        self.image_depth = image_depth
        self.transform = transform
        self.image_paths = self.read_folder()

    def read_folder(self): # input_folder에 있는 mp4 파일의 경로를 다 읽어와서 image_paths라는 빈 리스트에 추가
        '''Reads all the image paths in the given folder.
        '''
        image_paths = []
        for x in glob.glob(self.input_folder + '**'):
            if not x.endswith('mp4'):
                continue
            image_paths.append(x)
        return image_paths

    def __len__(self): # 위 함수에서 반환한 리스트의 길이 반환 = 데이터의 총 길이 반환
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''Returns a single image array.
        '''
        frames = []
        if torch.is_tensor(idx):        # tensor형태면 리스트 형태로 변경
            idx = idx.tolist()
        image = self.image_paths[idx]   # image 변수에 read_folder에서 반환한 input_folder에 있는 파일의 경로가 있는 리스트에서 하나씩(idx) 저장
        cap = cv2.VideoCapture(image)

        ##### 여기서 부터는 원래 동영상 읽어오려고 추가한 코드 #####
        for img in range(args["FPS"]):
            # while True:
            ret, frame = cap.read()
            if not ret:
                continue
            elif self.image_depth == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, args["img_size"])
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            else:
                frame = cv2.resize(frame, args["img_size"])
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            cap.release()
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)


def viz(args, device="cuda:0"):
    model_save_folder = './trained_model_I3D/'
    model = I3D_with_CBAM(num_classes=args["num_class"])

    model = model.to(device)

    assert os.path.exists(f"{model_save_folder}UBI-FIGHTS-VIDEO/I3D_cbam_UBI-FIGHTS-VIDEO_224x224_model_early.pth"), 'A trained model does not exist!'

    try:
        state_dict = torch.load(f"{model_save_folder}UBI-FIGHTS-VIDEO/I3D_cbam_UBI-FIGHTS-VIDEO_224x224_model_early.pth", map_location=device) # 가중치 불러옴, 딕셔너리 형태로 저장됨
        new_state_dict = OrderedDict() # 가중치에서 불필요한 부분 제거 후 필요한 것만 저장, 딕셔너리인데 순서까지 정렬

        for k,v in state_dict.items():  # new_state_dict의 딕셔너리 안의 모든 값을 k, v에 불러옴
            name = k[7:]  # state_dict 안에 있는 각 weight 값의 이름에서 앞의 일곱 글자를 제외한 나머지가 저장되는데, 이유는 이 모델이 저장될 때 'model.'로 시작됐기 때문에 모델 가중치 저장할때도 앞 7글자는 제외하고 저장되었기 때문이다. 앞 일곱 글자를 제외한 나머지를 name 변수에 저장
            new_state_dict[name] = v   # new_state_dict 변수에는 새로운 모델의 가중치 값을 저장할 딕셔너리 객체를 생성한다. 그리고 name 변수에 저장된 key 값과, state_dict에 저장된 value 값을 new_state_dict에 저장한다.

        model.load_state_dict(new_state_dict)  # 위의 new_state_dict를 model 객체에 할당
        print("Model loaded!")
    except Exception as e:
        print(e)

    model.eval()

    input_data = LoadInputImages(input_folder=path, image_size=args["img_size"], image_depth=args["img_depth"])  # transforms.ToTensor()는 이미지를 파이토치의 텐서 형식으로 변환하는 함수이다. # , transform=transforms.ToTensor()
    data_generator = DataLoader(input_data, batch_size=1, shuffle=False, num_workers=1)  # 입력 데이터를 배치 단위로 불러오기 위함임. DataLoader() 함수를 이용해 input_data를 배치 단위로 나누고, data_generator 변수에 저장함.

    # class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    # class_names = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
    class_names = ['fight', 'normal']  # ubi fights
    # class_names = ['Fight', 'NonFight']  # rwf 2000

    output_folder = './output_I3D/UBI_FIGHTS/'

    if not os.path.exists(output_folder) : os.mkdir(output_folder)


    fig = plt.figure(figsize=(10, 4))

    for i, image in tqdm(enumerate(data_generator)):

        plt.clf() # 그림 지움

        image = image.to(device)
        print(image.shape)
        # cnn_filters, output = model(image)  # model(image)의 결과가 cnn 필터와 model output 두 개임
        cnn_filters = model(image)  # model(image)의 결과가 cnn 필터와 model output 두 개임
        output = cnn_filters


        #identify the predicted class
        # softmaxed_output = torch.nn.Softmax(dim=1)(output)  # 출력값을 소프트맥스 함수에 적용하여 확률값으로 변환한다
        softmaxed_output = torch.nn.Softmax(dim=1)(output)  # 출력값을 소프트맥스 함수에 적용하여 확률값으로 변환한다
        predicted_class = class_names[torch.argmax(softmaxed_output).cpu().numpy()]  # 가장 높은 확률값을 가진 클래스를 예측한다.


        # 모든 필터를 하나로 병합하고 볼 수 있도록 원래 이미지 크기로 조정.
        # attention_combined_filter = cv2.resize(torch.max(attention_filters.squeeze(0), 0)[0].detach().numpy(), (args.img_size, args.img_size))
        cnn_combined_filter = cv2.resize(torch.max(cnn_filters.squeeze(0), 0)[0].detach().cpu().numpy(), (args["img_size"]))  # 위에서 구한 cnn_filters에서 가장 강한 부분 추출하여 필터를 병합하고 그 필터의 크기 리사이징함
        heatmap = np.asarray(cv2.applyColorMap(cv2.normalize(cnn_combined_filter, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET), dtype=np.float32) # 위에서 병합한 CNN 필터를 적용한 히트맵을 생성한다. cv2.normalize() 함수를 이용해 cnn_combined_filter 값의 범위를 0~255 사이로 정규화하고, cv2.COLORMAP_JET 컬러 맵을 적용하여 컬러 이미지로 변환한다cv2.applyColorMap() 함수를 이용해 히트맵을 생성하는데 이를 위해 컬러맵 적용.
        heatmap = heatmap / 255.

        print('heatmap.shape: ', heatmap.shape, heatmap.dtype)
        print('1', np.array(image.detach().cpu().numpy()).shape)
        print('2', np.array(image.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()).shape)
        print('2', np.array(image.squeeze(0).squeeze(1).detach().cpu().numpy()).shape)
        print('3', np.array(image.squeeze(-1).detach().cpu().numpy()).shape)
        # input_img = cv2.resize(image.squeeze(0).cpu().numpy(), (args["img_size"]))
        # input_img = cv2.resize(image.squeeze(0).cpu().numpy(), tuple(reversed(args["img_size"])))
        # input_img = cv2.resize(image.squeeze(0).cpu(), tuple(reversed(args["img_size"])))
        # input_img = cv2.resize(image.squeeze(0).squeeze(1).cpu().numpy(), (args["img_size"]))
        # input_img = image.squeeze(0).squeeze(1).permute(2, 1, 0).cpu().numpy()
        input_img = cv2.resize((image.squeeze(0).squeeze(1).permute(2, 1, 0).detach().cpu().numpy().astype(np.uint8)), dsize=args["img_size"])  # squeeze(0)을 이용해 차원을 축소하고, permute() 함수를 이용해 차원 순서를 변경해 입력 이미지 리사이즈

        print('input_img.shape: ', input_img.shape, input_img.dtype)
        # input_img = (input_img * 255).astype(np.uint8)
        input_img = input_img / 255.
        # input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)  # RGB 이미지를 그레이스케일로 변환

        # heatmap_cnn = cv2.addWeighted(np.asarray(input_img, dtype=np.float32), 0.9, heatmap.astype(np.float32), 0.0025, 0)  # 이때 cv2.addWeighted 함수를 사용하여 두 이미지를 가중치를 더하여 더한 결과를 얻는다
        heatmap_cnn = cv2.addWeighted(np.asarray(input_img, dtype=np.float32), 0.9, heatmap, 0.0025, 0)  # 이때 cv2.addWeighted 함수를 사용하여 두 이미지를 가중치를 더하여 더한 결과를 얻는다


        fig.add_subplot(131)
        plt.imshow(input_img)
        plt.title("Input Image")
        plt.xticks(())
        plt.yticks(())

        fig.add_subplot(132)
        plt.imshow(cnn_combined_filter)

        plt.xticks(())
        plt.yticks(())

        fig.add_subplot(133)
        plt.imshow(heatmap_cnn)
        plt.title("Heat Map")
        plt.xticks(())
        plt.yticks(())

        fig.suptitle(f"Network's prediction : {predicted_class.capitalize()}", fontsize=20)

        plt.savefig(f'{output_folder}/{i}.png')

if __name__ == '__main__':
    viz(args=args, device=device)