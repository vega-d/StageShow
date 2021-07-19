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
    ap.add_argument("-c", "--confidence", type=float, default=0.8,
                    help="minimum probability to filter weak detections. Default is 0.8")
    ap.add_argument("-t", "--precision", type=float, default=0.8,
                    help="How close to center of frame face should be? 0.1-0.9, Bigger number is closer to center. Default is 0.8")
    ap.add_argument("-a", "--anonymous", type=bool, default=0,
                    help="enable blanking out detected faces to avoid flashing your face.")
    ap.add_argument("-vh", "--visor_height", type=int, default=150,
                    help="Height in pixels for resolution in which webcam will be processed. Default is 500")
    ap.add_argument("-oh", "--output_height", type=int, default=480,
                    help="Height in pixels for resolution which will be used for output. Default is 360")
    ap.add_argument("-ow", "--output_width", type=int, default=854,
                    help="Width in pixels for resolution which will be used for output. Default is 640")
    ap.add_argument("--debug", help="Enable debug mode.", action="store_true", default=False)

    args = vars(ap.parse_args())

    return args


def distance_from(point1, point2):
    w_diff, h_diff = abs(point2[0] - point1[0]), abs(point2[1] - point1[1])
    return np.sqrt(w_diff ** 2 + h_diff ** 2)


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
    process2render_coefficient = resize / height
    renderframe_size = [args["output_width"], args["output_height"]]
    imagemagicsize = [int(i * process2render_coefficient) for i in renderframe_size]
    tolerance = float(args["precision"] / 2)
    cam = pyvirtualcam.Camera(width=renderframe_size[0], height=renderframe_size[1], fps=60)
    print("[INFO] Output virtual webcam is:", cam.device)
    print("[INFO] To test view the output execute: ffplay", cam.device)
    print("[INFO] Anonymous mode is", "on" if anon else "off")
    print("[INFO] Debug mode is", "on" if debug else "off")
    print("[INFO] process to render quality coefficient is", process2render_coefficient)

    screenboxX, screenboxY = width // 2 - imagemagicsize[0] // 2, height // 2 - imagemagicsize[1] // 2

    # loop over the frames from the video stream
    while True:
        # read the next frame from the video stream and resize it
        detect_frame = vs.read()[1]
        render_frame = detect_frame[:]
        detect_frame = imutils.resize(detect_frame, height=resize)
        if debug:
            debug_frame = detect_frame[:]


        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = detect_frame.shape[:2]

        # construct a blob from the frame, pass it through the network,
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        blob = cv2.dnn.blobFromImage(detect_frame, 1.0, (W, H), (104.0, 177.0, 123.0))
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
                    points = np.array([[startX - 20, startY - 20], [endX + 20, startY - 20], [endX + 20, endY + 20],
                                       [startX - 20, endY + 20]], dtype=np.int32)
                    cv2.fillPoly(detect_frame, [points], (0, 0, 0))

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = list(ct.update(rects).items())

        if len(objects):
            tracking_object = objects[0]

            objectID, centroid = tracking_object

            if anon:
                cv2.putText(detect_frame, "anonymous mode on", (centroid[0] - 85, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

            speed = int(distance_from(centroid,
                                      (screenboxX + imagemagicsize[0] // 2, screenboxY + imagemagicsize[1] // 2)) // 30)
            # speed = 10

            if centroid[0] < (screenboxX + int(imagemagicsize[0] * tolerance)):
                screenboxX -= speed
            if centroid[0] > (screenboxX + int(imagemagicsize[0] * (1 - tolerance))):
                screenboxX += speed

            if centroid[1] < (screenboxY + int(imagemagicsize[1] * tolerance)):
                screenboxY -= speed
            if centroid[1] > (screenboxY + int(imagemagicsize[1] * (1 - tolerance))):
                screenboxY += speed

            # print(speed, screenboxX, screenboxY)

            if debug:
                cv2.rectangle(detect_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

                text1 = str(imagemagicsize)
                cv2.putText(detect_frame, text1, (screenboxX + 20, screenboxY + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

                cv2.rectangle(detect_frame, (
                    screenboxX + int(imagemagicsize[0] * tolerance), int(screenboxY + imagemagicsize[1] * tolerance)),
                              (screenboxX + int(imagemagicsize[0] * (1 - tolerance)),
                               screenboxY + int(imagemagicsize[1] * (1 - tolerance))), (255, 0, 255), 2)

                text = ("occ: " + str(debug))
                cv2.putText(detect_frame, text, (screenboxX + 20, screenboxY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

            screenboxX, screenboxY = min(max(screenboxX, 0), W - imagemagicsize[0]), min(max(screenboxY, 0),
                                                                                         H - imagemagicsize[1])

            if debug:
                # debug_frame = detect_frame[:]
                cv2.rectangle(debug_frame, (screenboxX, screenboxY),
                              (screenboxX + imagemagicsize[0], screenboxY + imagemagicsize[1]), (0, 0, 255), 2)
        else:
            # frame = frame[0:imagemagicsize[1], 0:imagemagicsize[0]]
            text = "No face found !"
            cv2.putText(detect_frame, text, (screenboxX + 20, screenboxY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        renderX, renderY = int(screenboxX / process2render_coefficient), int(screenboxY / process2render_coefficient)

        render_frame = render_frame[renderY:renderY + renderframe_size[1], renderX:renderX + renderframe_size[0]]
        if debug:
            cv2.imshow(cam.device, debug_frame)

        # render_frame = render_frame[0:renderframe_size[1], 0:renderframe_size[0]]
        render_frame = imutils.resize(render_frame, height=args["output_height"], width=args["output_width"])
        render_frame = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
        cam.send(render_frame)

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
