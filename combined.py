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
import numpy as np
from utils import YOLOv5s
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
parser.add_argument("--wb", "-b", type=int, default=10, help="Weight of basketball")
parser.add_argument("--wp", "-p", type=int, default=7, help="Weight of player")
args = parser.parse_args()

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

# Threading Variable
q = queue.Queue()
timestamp = "a"


# Starting the Client
async def client():
    uri = "ws://192.168.147.222:6969/"  # Use your server's IP address and port
    flag = False

    global timestamp

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

            index = 0
            filename = f'{PATH}/{timestamp}/video_{index}.mp4'

            # Running Inference and Storing it directly
            logger.info("Opening stream on device: {}".format(args.device))

            start_time = chunk_time = time.time()

            writer = cv2.VideoWriter(filename, fourcc, fps, resolution)

            cumulative_weight = 0

            # Entire lenght of video
            while time.time() - start_time < args.time*2:
                try:
                    # Check for response from the server and act accordingly
                    try:
                        server_response = await asyncio.wait_for(websocket.recv(), timeout=0.000001)
                        decision = server_response.split("_")[1]
                        if decision == "upload" or decision == "delete":
                            q.put(server_response)
                        server_response = "placeholder_placeholder"
                    except asyncio.TimeoutError:
                        pass

                    res, frame = camera.read()
                    if res is False:
                        logger.error("Empty image received")
                        break

                    else:
                        # Run Inference
                        input_frame = model.preprocess_frame(frame)
                        output = model.inference(input_frame)
                        detections = model.postprocess(output)
                        output_frame, frame_weight = model.draw_bbox_weights(frame, detections, args.wb, args.wp)

                        cumulative_weight += frame_weight

                        writer.write(output_frame)

                        # s = ""
                        #
                        # for c in np.unique(detections[:, -1]):
                        #     n = (detections[:, -1] == c).sum()
                        #     s += f"{n} {classes[int(c)]}{'s' * (n > 1)}, "
                        #
                        # if s != "":
                        #     s = s.strip()
                        #     s = s[:-1]

                        # logger.info("Detected: {}".format(s))

                        if time.time() - chunk_time >= 17:
                            writer.release()  # Save Video chunk

                            # Send cumulative weight
                            await websocket.send(f"{index}_{cumulative_weight}")

                            index += 1
                            cumulative_weight = 0  # Reset weights for next chunk
                            filename = f'{PATH}/{timestamp}/video_{index}.mp4'  # Update filename index
                            writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
                            chunk_time = time.time()

                        cv2.waitKey(1)

                except KeyboardInterrupt:
                    break

            else:
                writer.release()

                filename = f'{PATH}/{timestamp}/video_{index}.mp4'

                # Send cumulative weight for last frame
                await websocket.send(f"{index}_{cumulative_weight}")

                while True:
                    try:
                        server_response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        decision = server_response.split("_")[1]
                        if decision == "upload" or decision == "delete":
                            q.put(server_response)
                        server_response = "placeholder_placeholder"
                    except asyncio.TimeoutError:
                        pass

            camera.release()
            cv2.destroyAllWindows()
            q.put(f"1_end")
            flag = False


#def main_thread():
#    asyncio.get_event_loop().run_until_complete(client())

def main_thread():
    loop = asyncio.new_event_loop()  # Create a new event loop
    asyncio.set_event_loop(loop)     # Set it as the current event loop
    loop.run_until_complete(client())  # Run the coroutine

def upload_thread():
    global q
    global timestamp

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
                logger.info("Script Ended")
                break
            elif decision == "upload": 
                # Upload video_path (local) to database_path (database)
                database_path = f"{timestamp}/video_{index}"
                bucket = storage.bucket()
                blob = bucket.blob(database_path)
                blob.upload_from_filename(video_path)

                # optional
                blob.make_public()
                print(f"video_{index}'s url: ", blob.public_url)

                # we can delete the video from the local storage
                # afterwards if needed
            elif decision == "delete":
                if os.path.exists(video_path):
                    os.remove(video_path)


if __name__ == "__main__":

    t1 = threading.Thread(target=main_thread)
    t2 = threading.Thread(target=upload_thread)
    while True:
        t1.start()
        t2.start()

        t1.join()
        t2.join()
