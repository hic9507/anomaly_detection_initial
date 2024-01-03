import random
import shutil
import os

nonvi = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos/videos/train/normal/'
vi = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos/videos/train/fight/'

nonvi_test = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos/videos/val/normal/'
vi_test = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos/videos/val/fight/'

num_nonvi = int(len(os.listdir(nonvi)) * 0.2)
num_vi = int(len(os.listdir(vi)) * 0.2)

# num_nonvi = int(len(glob.glob(nonvi + '/*' + '/*')) * 0.2)
# num_vi = int(len(glob.glob(vi + '/*' + '/*')) * 0.2)
print(num_nonvi, num_vi) # 156 43

# for i in range(num_nonvi):
#     # src =
#     print(os.listdir(nonvi)[0])
#     break
# sample_list0 = random.sample(os.listdir(nonvi), num_nonvi) # train normal
# print(len(sample_list0))
# for i in sample_list0:
#     src = nonvi + i
#     dst = nonvi_test + i
#     shutil.move(src, dst)

sample_list1 = random.sample(os.listdir(vi), num_vi) # train fight
print(len(sample_list1))
for i in sample_list1:
    src = vi + i
    dst = vi_test + i
    shutil.move(src, dst)


########## 폴더를 통으로 옮기기 시작 ##########
# sample_list = random.sample(os.listdir(nonvi), num_nonvi)
# for i in sample_list:
#     src = nonvi + i
#     dst = 'D:/InChang/SCVD/frames_new_split_test/NonViolence/' + i
#     shutil.move(src, dst)

# sample_list = random.sample(os.listdir(vi), num_vi)
# for i in sample_list:
#     src = vi + i
#     dst = 'D:/InChang/SCVD/frames_new_split_test/NonViolence/' + i
#     shutil.move(src, dst)
########## 폴더를 통으로 옮기기 끝 ##########
# nonvi_list, vi_list, nonvi_test_list, vi_test_list = [], [], [], []
# for i in glob.glob(nonvi + '/*' + '/*'):
#     nonvi_list.append(i)
# for i in glob.glob(vi + '/*' + '/*'):
#     vi_list.append(i)
# for i in glob.glob(nonvi_test + '/*' + '/*'):
#     nonvi_test_list.append(i)
# for i in glob.glob(vi_test + '/*' + '/*'):
#     vi_test_list.append(i)

# print(len(nonvi_list), len(vi_list), len(nonvi_test_list), len(vi_test_list))

# sample_list = random.sample(glob.glob(nonvi + '/*' + '/*', num_nonvi))
# for i in sample_list:
#     src = vi + i
#     dst = 'D:/InChang/SCVD/frames_new_split_test/NonViolence/' + i
#     shutil.move(src, dst)

# sample_list = random.sample(glob.glob(vi + '/*' + '/*', num_vi))
# for i in sample_list:
#     src = vi + i
#     dst = 'D:/InChang/SCVD/frames_new_split_test/Violence/' + i
#     shutil.move(src, dst)


# for i in range(num_nonvi):
# sample_list = random.sample(os.listdir(vi), num_vi)
# for i in sample_list:
#     src = vi + i
#     dst = 'D:/InChang/SCVD/frames_new_split_test/Violence/' + i
#     shutil.move(src, dst)
