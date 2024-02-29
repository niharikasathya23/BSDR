import cv2
import depthai as dai
import os
import threading
from queue import Queue

# Output directories
output_dir_left = 'left_images'
output_dir_right = 'right_images'
os.makedirs(output_dir_left, exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)

num_images = 20

pipeline = dai.Pipeline()

# Setup mono cameras
left = pipeline.createMonoCamera()
right = pipeline.createMonoCamera()
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Setup XLinkOut
xout_left = pipeline.createXLinkOut()
xout_right = pipeline.createXLinkOut()
xout_left.setStreamName("left")
xout_right.setStreamName("right")

left.out.link(xout_left.input)
right.out.link(xout_right.input)

# Image save queue and stop signal
save_queue = Queue()
stop_signal = threading.Event()

def save_image():
    while not stop_signal.is_set() or not save_queue.empty():
        item = save_queue.get()
        if item is None:  # Check for the stop signal
            break
        cv2.imwrite(*item)
        print(f"Image saved as {item[0]}")
        save_queue.task_done()

# Start the save_image thread
thread = threading.Thread(target=save_image, daemon=True)
thread.start()

def main():
    with dai.Device(pipeline) as device:
        leftQueue = device.getOutputQueue(name="left", maxSize=4, blocking=True)
        rightQueue = device.getOutputQueue(name="right", maxSize=4, blocking=True)

        image_count = 0
        while image_count < num_images:
            inLeft = leftQueue.get()
            inRight = rightQueue.get()

            frame_left = inLeft.getCvFrame()
            frame_right = inRight.getCvFrame()

            cv2.imshow('Left Image', cv2.resize(frame_left, (640, 360)))
            cv2.imshow('Right Image', cv2.resize(frame_right, (640, 360)))

            key = cv2.waitKey(1)
            if key == ord('c'):
                left_filename = os.path.join(output_dir_left, f'left_{image_count:03d}.png')
                right_filename = os.path.join(output_dir_right, f'right_{image_count:03d}.png')
                save_queue.put((left_filename, frame_left))
                save_queue.put((right_filename, frame_right))
                image_count += 1
            elif key == ord('q'):
                break

# Call the main function
if __name__ == "__main__":
    try:
        main()
    finally:
        stop_signal.set()
        save_queue.put(None)  
        thread.join()
        cv2.destroyAllWindows()
