import argparse
import cv2
import glob

# export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0

def create_timelapse(result_dir, frame_rate, width, height):
    images = sorted(glob.glob(f'{result_dir}/Generated_Image/*.jpg'))

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(f'{result_dir}/timelaps.mp4', fourcc, frame_rate, (width, height))
    for i in range(len(images)):
        img = cv2.imread(images[i])
        img = cv2.resize(img,(width,height))
        video.write(img) 
    video.release()
    print(f"Timelapse video saved as {result_dir}/timelaps.mp4")

def main():
    parser = argparse.ArgumentParser(description="Create a timelapse video from images.")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to the folder containing the images.")
    parser.add_argument("--frame_rate", type=float, default=2, help="Frame rate of the output video.")
    parser.add_argument("--width", type=int, default=1042, help="Width of the output video.")
    parser.add_argument("--height", type=int, default=2082, help="Height of the output video.")

    args = parser.parse_args()
    create_timelapse(args.result_dir, args.frame_rate, args.width, args.height)

if __name__ == "__main__":
    main()
