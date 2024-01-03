import os
import cv2


def check_video_file(video_path):
    # 비디오 파일 열기
    video = cv2.VideoCapture(video_path)

    # 파일이 제대로 열렸는지 확인
    if not video.isOpened():
        print(f"Could not open video file: {video_path}", '*'*50)
        return False

    # 비디오 프레임을 순차적으로 읽기
    while True:
        # 프레임 읽기
        ret, frame = video.read()

        # ret는 프레임을 성공적으로 읽었는지를 나타냄 (True/False)
        if not ret:
            break

    # 비디오 파일 닫기
    video.release()

    print(f"Video file {video_path} was read successfully.")
    return True


def check_videos_in_directory(directory):
    cnt = 0
    # 디렉토리 안의 모든 파일과 하위 디렉토리 탐색
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 파일의 확장자 확인
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):  # 추가적인 비디오 형식이 필요하면 확장자를 추가
                # 파일 경로 생성
                video_path = os.path.join(root, file)
                # 비디오 파일 체크 함수 호출
                check_video_file(video_path)
                cnt += 1
            print(cnt)


# 사용할 폴더 경로
directory_path = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos/videos/'
check_videos_in_directory(directory_path)