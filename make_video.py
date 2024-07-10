import cv2
import os

dir_name = "generations/"
video_dir = "video.mp4"
video_length = 50 # Length of the video (in seconds)


def toVid():
    print("Writing video...")
    dir_list = os.listdir(dir_name)
    height, width, layers = cv2.imread(dir_name + dir_list[0]).shape
    fps = round(len(dir_list) / video_length)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_dir, fourcc, fps, (width, height))

    for x in range(len(os.listdir(dir_name))):
        video.write(cv2.imread(dir_name + str(x+1) + ".jpg"))

    cv2.destroyAllWindows()
    video.release()

    print(f"Saved video as {video_dir}")


if __name__ == '__main__':
    toVid()
