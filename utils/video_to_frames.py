import os
import cv2


def video_to_frames(video_path, output_path):
    """Given a video, extract frames from it and save them as images.
    Usage: python video_to_frames.py <video-path> <path-to-save-frames>
    e.g. python video_to_frames.py myvideo.avi myframes/
    video_path: path to video file
    output_path: path to save frames
    """

    # Playing video from file:
    video = cv2.VideoCapture(video_path)

    # get name of video file
    video_name = os.path.basename(video_path)
    # get name of video file without extension
    video_name = os.path.splitext(video_name)[0]
    output_path = os.path.join(output_path, video_name, 'frames')

    # Create output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    current_frame = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        if current_frame % 100 == 0:
            print(f"Processing frame {current_frame}")
        if ret:
            # Saves image of the current frame in jpg file
            name = os.path.join(output_path, str(current_frame) + '.jpg')
            cv2.imwrite(name, frame)
            # To stop duplicate images
            current_frame += 1
        else:
            break

    print(f"Processed {video_path} and saved frames to {output_path}")


def frames_to_video(input_folder, output_video_path, fps):
    # Get the list of frames
    frames = [img for img in os.listdir(input_folder) if img.endswith(".jpg")]

    # Sort the frames by frame number to ensure correct order
    order = [int(frame.split(".")[0]) for frame in frames]
    frames = [frame for _, frame in sorted(zip(order, frames))]

    # Determine the size of the frames
    frame = cv2.imread(os.path.join(input_folder, frames[0]))
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can change the codec as needed
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Add frames to the video
    for frame in frames:
        print(f"Adding frame {frame} to video {output_video_path}")
        img_path = os.path.join(input_folder, frame)
        img = cv2.imread(img_path)
        video.write(img)

    # Release the video writer
    video.release()
