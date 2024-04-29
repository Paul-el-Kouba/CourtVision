import os
import cv2
import time
import queue
import random
import asyncio
import logging
import argparse
import threading
import websockets
import subprocess
import numpy as np
from PID import PID
from utils import YOLOv5s
from periphery import PWM
from datetime import datetime
from firebase_admin import credentials, initialize_app, storage

PATH = os.getcwd()

# Managing Arguments
parser = argparse.ArgumentParser("EdgeTPU test runner")

parser.add_argument("--model", "-m", help="Weights file", required=True)
parser.add_argument("--labels", "-l", type=str, required=True, help="Labels file")
parser.add_argument("--device", "-dev", type=int, default=1,
                    help="Camera to process feed from (0, for Coral Camera, 1 for USB")
parser.add_argument("--time", "-t", type=int, default=300, help="Length of video to record")
parser.add_argument("--conf", "-ct", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--iou", "-it", type=float, default=0.1, help="Detections IOU threshold")
parser.add_argument("--gray", "-g", help="Normal or Grayscale")

args = parser.parse_args()

controller = PID(0.000015, 0, 0.000001)
pwm = PWM(1, 0)
pwm.frequency = 50
pwm.duty_cycle = 0.9
pwm.enable()

time.sleep(1)

# Setting up Database
cred = credentials.Certificate("./serviceAccountKey.json")
initialize_app(cred, {'storageBucket': 'database-c8de2.appspot.com'})

# Loading Detection Model and Classes
model = YOLOv5s(args.model, args.labels, args.conf, args.iou)
classes = model.load_classes(args.labels)

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeTPUModel")
logger.info("Loaded {} classes".format(len(classes)))

# Setting up the Camera
resolution = (640, 480)
fps = 30

# Initialize the ArduCam USB camera
camera = cv2.VideoCapture(args.device)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
camera.set(cv2.CAP_PROP_FPS, fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize Variables
q = queue.Queue() #upload-delete
streamq = queue.Queue()
timestamp = "a"
decision = "b"

frame_q = queue.Queue()
endInference = False

m_or_s = False
switch = False

startFilming = False

main_cam = False

async def client():
    uri = "ws://172.20.10.3:6969/"  # Use your server's IP address and port
    flag = False

    global main_cam
    global startFilming
    global m_or_s

    async with websockets.connect(uri) as websocket:
        # Send to server so it knows a client is connected
        await websocket.send("0")

        # Keep the connection open until a response is received
        while True:
            try:
                # Receive the response from the server
                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                if response == "start":
                    flag = True
                    break
            except asyncio.TimeoutError:
                # If no response is received within the timeout, keep waiting
                pass

        # Once "start" from the server is received, enter this loop and start working
        while flag:
            timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            os.mkdir(f"{PATH}/{timestamp}")

            # Running Inference and Storing it directly
            logger.info("Opening stream on device: {}".format(args.device))
            startFilming = True

            while startFilming:
                if frame_q.empty():
                    continue
                else:
                    frame = frame_q.get()

                    input = model.preprocess_frame(frame)

                    output = model.inference(input)

                    detections = model.postprocess(output)

                    # s = ""
                    #
                    # for c in np.unique(detections[:, -1]):
                    #     n = (detections[:, -1] == c).sum()
                    #     s += f"{n} {classes[int(c)]}{'s' * (n > 1)}, "
                    #
                    # if s != "":
                    #     s = s.strip()
                    #     s = s[:-1]
                    #
                    # logger.info("Detected: {}".format(s))

                    if len(detections) >= 1:
                        main_cam = True
                        center_frame = frame.shape[1] / 2

                        center_obj = (detections[0][0] + detections[0][2]) / 2

                        error = center_obj - center_frame

                        corr = controller(error)

                        pwm.duty_cycle = np.clip(pwm.duty_cycle + corr, 0.865, 0.965)
                        print(corr, error, pwm.duty_cycle)
                    else:
                        main_cam = False


def main_thread():
    loop = asyncio.new_event_loop()  # Create a new event loop
    asyncio.set_event_loop(loop)  # Set it as the current event loop
    loop.run_until_complete(client())  # Run the coroutine


def upload_thread():
    global q
    global timestamp
    global m_or_s

    while True:

        if q.empty():
            continue
        else:
            response = q.get().split("_")
            index = response[0]
            decision = response[1]

            print(decision)

            video_path = f"{PATH}/{timestamp}/video_{index}.mp4"

            if decision == "end":
                streamq.put("end")
                logger.info("Script Ended")
                break
            elif main_cam:
                # Upload video_path (local) to database_path (database)
                streamq.put(video_path)
                database_path = f"{timestamp}/video_{index}"
                bucket = storage.bucket()
                blob = bucket.blob(database_path)
                blob.upload_from_filename(video_path)

                # optional
                blob.make_public()
                print(f"video_{index}'s url: ", blob.public_url)

                # we can delete the video from the local storage
                # afterward if needed
            elif not main_cam:
                if os.path.exists(video_path):
                    os.remove(video_path)


def streaming_thread():
    global streamq
    global timestamp
    while True:
        if streamq.empty():
            continue
        else:
            vid = streamq.get()
            if vid == "end":
                return
            else:
                ffmpeg_command = [
                    "ffmpeg",
                    "-re",
                    "-i",
                    vid,
                    "-map", "0:v",
                    "-an",
                    "-c:v", "libvpx",
                    "-b:v", "1M",
                    "-f", "rtp", "rtp://172.20.10.4:5105"
                ]
                subprocess.run(ffmpeg_command)


def tracking_thread():

    while True:
        if startFilming:
            index = 0
            filename = f'{PATH}/{timestamp}/video_{index}.mp4'

            start_time = chunk_time = time.time()

            writer = cv2.VideoWriter(filename, fourcc, fps, resolution)

            count = 0

            while time.time() - start_time < args.time:
                try:
                    # Check for response from the server and act accordingly
                    res, frame = camera.read()
                    if res is False:
                        logger.error("Empty image received")
                        break

                    else:

                        if count % 4 == 0:
                            frame_q.put(frame)
                        count += 1

                        writer.write(frame)

                        if time.time() - chunk_time >= 2 or switch:
                            if switch:
                                switch = False
                            writer.release()  # Save Video chunk

                            index += 1
                            filename = f'{PATH}/{timestamp}/video_{index}.mp4'  # Update filename index
                            writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
                            chunk_time = time.time()

                        cv2.waitKey(1)
                except KeyboardInterrupt:
                    break
            break

if __name__ == "__main__":

    t1 = threading.Thread(target=main_thread)
    t2 = threading.Thread(target=upload_thread)
    t3 = threading.Thread(target=streaming_thread)
    t4 = threading.Thread(target=tracking_thread)
    while True:
        t1.start()
        t2.start()
        t3.start()
        t4.start()

        t1.join()
        t2.join()
        t3.join()
        t4.join()
