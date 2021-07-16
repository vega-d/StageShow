# Copyright https://github/vega-d 2021
# import the necessary packages
from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import pyvirtualcam


def argsParser():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file", default="./model/deploy.prototxt")
    ap.add_argument("-m", "--model", required=False,
                    help="path to Caffe pre-trained model", default="./model/res10_300x300_ssd_iter_140000.caffemodel")
    ap.add_argument("-c", "--confidence", type=float, default=0.9,
                    help="minimum probability to filter weak detections. Default is 0.9")
    ap.add_argument("-a", "--anonymous", type=bool, default=0,
                    help="enable blanking out detected faces to avoid flashing your face.")
    ap.add_argument("-vh", "--visor_height", type=int, default=500,
                    help="Height in pixels for resolution in which webcam will be processed. Default is 500")
    ap.add_argument("-oh", "--output_height", type=bool, default=360,
                    help="Height in pixels for resolution which will be used for output. Default is 360")
    ap.add_argument("-ow", "--output_width", type=bool, default=640,
                    help="Width in pixels for resolution which will be used for output. Default is 640")
    ap.add_argument("--debug", help="Enable debug mode.", action="store_true", default=False)

    args = vars(ap.parse_args())

    return args


def run_face_tracker(args):
    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    (H, W) = (None, None)

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    HIGH_VALUE = 10000
    WIDTH = HIGH_VALUE
    HEIGHT = HIGH_VALUE

    vs = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

    time.sleep(0.2)
    print("[INFO] Started video stream.")
    anon = args["anonymous"]
    debug = args["debug"]
    resize = args["visor_height"]
    imagemagicsize = [args["output_width"], args["output_height"]]

    cam = pyvirtualcam.Camera(width=imagemagicsize[0], height=imagemagicsize[1], fps=60)
    print("[INFO] Output virtual webcam is:", cam.device)
    print("[INFO] To test view the output execute: ffplay" + cam.device)
    print("[INFO] Anonymous mode is", "on" if anon else "off")
    print("[INFO] Debug mode is", "on" if debug else "off")

    # loop over the frames from the video stream
    while True:
        # read the next frame from the video stream and resize it
        frame = vs.read()[1]
        frame = imutils.resize(frame, height=resize)

        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the frame, pass it through the network,
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            if detections[0, 0, i, 2] > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))

                # draw a bounding box surrounding the object so we can
                # visualize it
                (startX, startY, endX, endY) = box.astype("int")

                if anon:
                    points = np.array([[startX - 20, startY - 20], [endX + 20, startY - 20], [endX + 20, endY + 20], [startX - 20, endY + 20]], dtype=np.int32)
                    cv2.fillPoly(frame, [points], (0, 0, 0))

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = list(ct.update(rects).items())

        if len(objects):
            tracking_object = objects[0]

            objectID, centroid = tracking_object

            valX = centroid[0] - int(imagemagicsize[0] * 0.5)
            valY = centroid[1] - int(imagemagicsize[1] * 0.5)

            if anon:
                cv2.putText(frame, "anonymous mode on", (centroid[0] - 85, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            screenboxX, screenboxY = min(max(valX, 0), W - imagemagicsize[0]), min(max(valY, 0),
                                                                                   H - imagemagicsize[1])
            frame = frame[screenboxY:screenboxY + imagemagicsize[1], screenboxX:screenboxX + imagemagicsize[0]]


        else:
            frame = frame[0:imagemagicsize[1], 0:imagemagicsize[0]]
            text = "No face found !"
            cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if debug:
            cv2.imshow(cam.device, frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cam.send(frame)

        cam.sleep_until_next_frame()
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def main():
    args = argsParser()
    run_face_tracker(args)


# Run face tracking
if __name__ == '__main__':
    main()
