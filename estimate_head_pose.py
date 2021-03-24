"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import cv2
import os
import numpy as np

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

print("OpenCV version: {}".format(cv2.__version__))

# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--out", type=str, default=None,
                    help="Video output path. ")
parser.add_argument("--input_path", type=str, default=None,
                    help="input videos path. ")
args = parser.parse_args()


def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)


def main():
    """MAIN"""
    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    if args.out != None:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        output_movie = cv2.VideoWriter(args.out, fourcc, 30, (width, height))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()

    cnt = 0

    input_path = args.input_path
    listdir = os.listdir(input_path)
    for v_name in listdir:
        v_path = os.path.join(input_path, v_name)
        cap = cv2.VideoCapture(v_path)

        while True:
            # Read frame, crop it, flip it, suits your needs.
            frame_got, frame = cap.read()
            if frame_got is False:
                break

            # Crop it if frame is larger than expected.
            # frame = frame[0:480, 300:940]

            # If frame comes from webcam, flip it so it looks like a mirror.
            if video_src == 0:
                frame = cv2.flip(frame, 2)

            # Pose estimation by 3 steps:
            # 1. detect face;
            # 2. detect landmarks;
            # 3. estimate pose

            # Feed frame to image queue.
            img_queue.put(frame)

            # Get face from box queue.
            facebox = box_queue.get()

            if facebox is not None:
                # Detect landmarks from image of 128x128.
                face_img = frame[facebox[1]: facebox[3],
                           facebox[0]: facebox[2]]
                face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                tm.start()
                marks = mark_detector.detect_marks(face_img)
                tm.stop()

                # Convert the marks locations from local CNN to global image.
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]

                # Uncomment following line to show raw marks.
                # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

                # Uncomment following line to show facebox.
                # mark_detector.draw_box(frame, [facebox])

                # Try pose estimation with 68 points.
                pose = pose_estimator.solve_pose_by_68_points(marks)

                # Stabilize the pose.
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))

                # Uncomment following line to draw pose annotation on frame.
                # pose_estimator.draw_annotation_box(
                #     frame, pose[0], pose[1], color=(255, 128, 128))

                # Uncomment following line to draw stabile pose annotation on frame.
                pose_estimator.draw_annotation_box(
                    frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

                # Uncomment following line to draw head axes on frame.
                # pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])

            # Show preview.
            # cv2.imshow("Preview", frame)
            # if cv2.waitKey(10) == 27:
            #     break
            if args.out != None:
                output_movie.write(frame)
            else:
                cv2.imshow("Preview", frame)

            cnt = cnt + 1
            if cnt % 100 == 0:
                print(str(cnt), flush=True)

    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
