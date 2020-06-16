# PianoTranscription.py
# John Weingart and Jack Williams
# March, 2020
#
# Determines the notes being played

import cv2
import numpy as np
import math
import time
import operator
import warnings
import mido
import aubio
from matplotlib import pyplot as plt
import time

SIGMA = 0.33 # indicates threshold for edge detection
LINE_DETECT_HEIGHT_PROP = 0.6
LINE_DETECT_WIDTH_PROP = 0.3
MIN_PIANO_PROP = 0.1
MAX_PIANO_PROP = 0.4
MAX_LINES = 20
FIND_PIANO_COUNT = 100
VID_FILE_NAME = "Crain.mp4"
MIDI_BUS_NAME = "IAC Driver pybus"
THRESH_MEAN_KEY = 2
THRESH_DIFF_PRESSED = 15
THRESH_ONSET = 1
P_FPS = 3
PIANO_DICT = {0:0, 1:2, 2:3, 3:5, 4:7, 5:8, 6:10}

# cropPianoVert(): Crops the top and bottom of the piano, using the first FIND_PIANO_COUNT
# frames of the video. Applies Canny Edge Detection and a Hough Line Transform on each frame,
# to detect the dominant edges of the top and bottom of the piano. Iterates through each pair of
# edges, to determine those with the brightest area in the bottom 1/3 of the area between them; this
# represents the keyboard.
# returns:
#       maximum_bottom: The best coordinate of the bottom of the keyboard
#       maximum_top: The best coordinate of the top of the keyboard

def cropPianoVert():
    vid = cv2.VideoCapture(VID_FILE_NAME)
    count = 0
    y_bottom_counts = dict()
    y_top_counts = dict()

    while count < FIND_PIANO_COUNT:
        _,frame = vid.read()
        if frame is None:
            break
        height, width = frame.shape[:2]
        v = np.median(frame)

        #set threshold values for Canny detection
        lower = int(max(0, (1.0 - SIGMA) * v))
        upper = int(min(255, (1.0 + SIGMA) * v))

        edges = cv2.Canny(frame, lower, upper)

        cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lines = cv2.HoughLines(edges, 1, 3.14 / 180, 80, None, 0, 0)
        curr_max = 0
        y_bottom = height
        y_top = 0
        if lines is not None:
            numLines = min(len(lines), MAX_LINES)
            for i in range(0, numLines):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + width * (-b)), int(y0 + height * a))
                pt2 = (int(x0 - width * (-b)), int(y0 - height * a))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

                for j in range(0, numLines):
                    if j is not i:
                        y1 = lines[j][0][0] * math.sin(lines[j][0][1])
                        if y0 > y1:
                            pianoImg = grayScale[int(y1 + (y0 - y1) * 2 / 3):int(y0), 0:width]
                        elif y1 > y0:
                            pianoImg = grayScale[int(y0 + (y1 - y0) * 2 / 3):int(y1), 0:width]
                        if np.median(pianoImg) > curr_max and math.fabs(y1-y0) > height*.1:
                            curr_max = np.median(pianoImg)
                            y_top = min(y0, y1)
                            y_bottom = max(y0, y1)
            if y_bottom in y_bottom_counts:
                y_bottom_counts[y_bottom] += 1
            else:
                y_bottom_counts.update({y_bottom: 1})
            if y_top in y_top_counts:
                y_top_counts[y_top] += 1
            else:
                y_top_counts.update({y_top: 1})
        cv2.imshow('lines', cdst)
        count += 1
    vid.release()
    maximum_bottom = max(y_bottom_counts.iteritems(), key=operator.itemgetter(1))[0]
    maximum_top = max(y_top_counts.iteritems(), key=operator.itemgetter(1))[0]

    return maximum_bottom, maximum_top

# cropPianoSide():
# Iteratively crops the left and right sides, one pixel at a time, until the area to be chopped is
# sufficiently bright, to represent a key on the keyboard
# returns:
#       countLeft: the amount of pixels to crop out on the left
#       countRight: the amount of pixels to crop out on the right
def cropPianoSide(image):
    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = image.shape[:2]
    whiteKeys = image[int(height*2/3):height, 0:width]
    thresh_crop = np.median(whiteKeys)
    print(thresh_crop)

    #count pixels to cut off left and right
    countLeft = 0
    countRight = 0;

    while True:
        height, width = image.shape[:2]
        sliceLeft = image[int(height*2/3):height, 0:1];
        print("thresh_crop", thresh_crop, "np median:", np.median(sliceLeft))
        if np.median(sliceLeft) < thresh_crop*.4 or np.median(sliceLeft) > thresh_crop*1.3:
            image = image[0:height, 1:width]
        else:
            break
        countLeft += 1
    while True:
        height, width = image.shape[:2]
        sliceRight = image[int(height*2/3):height, (width-1):width];
        if np.median(sliceRight) < thresh_crop*.4 or np.median(sliceRight) > thresh_crop*1.3:
            image = image[0:height, 0:(width-1)]
        else:
            break
        countRight += 1

    return countLeft, countRight

# baseline(bottom, top):
# Loops through all frames in the mp4 video, searching for one frame that does
# not contain hands. It searches using the mean brightness of the bottom 1/3 of
# the piano (where the black keys end). The frame with the maximum brightness is
# returned, to be used as a baseline frame for comparison.
# parameters:
#      bottom: the bottom of the image
#      top: the top of the image
# returns:
#      best_frame: the ideal frame to use as the baseline
def baseline(bottom, top):
    vid = cv2.VideoCapture(VID_FILE_NAME)
    #originally: set first frame as best frame
    _, first_frame = vid.read()
    height, width = first_frame.shape[:2]
    whiteKeys = first_frame[int(top + (bottom - top)*2/3):int(bottom), 0:width]
    grayScale = cv2.cvtColor(whiteKeys, cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    best_frame_brightness = np.mean(grayScale)
    best_frame = grayFrame

    while True:
        _, frame = vid.read()
        if frame is None:
            break
        height, width = frame.shape[:2]
        whiteKeys = frame[int(top+(bottom-top)*2/3):int(bottom), 0:width]
        grayScale = cv2.cvtColor(whiteKeys, cv2.COLOR_BGR2GRAY)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(grayScale)
        if brightness > best_frame_brightness:
            best_frame = grayFrame
            best_frame_brightness = brightness
    vid.release()
    return best_frame[int(top):int(bottom), 0:width]

# convertMIDI(lowest_key, i)
# Converts an integer index (representing a white key's position in the image) to a MIDI note
# number. Assumes that the lowest note in the image is an A.
# parameters:
#       lowest_key: The user-entered low key index on the piano
#       i: the index of the key to be converted to MIDI
# return:
#       MIDI: the MIDI value associated with the inputted key
def convertMIDI(lowest_key, i):
    i = i + lowest_key
    key = i % 7
    octave = int(i / 7);
    half_steps = PIANO_DICT[key]
    MIDI = half_steps + 12 * octave + 21
    return MIDI

# initAudio():
# Initializes the audio analysis tools, analyzes each sample in
# the audio, to generate a list of all onsets in the mp4.
# returns:
#       onsets: a list of all of the onsets in the frame
#       src.samplerate: the rate of sampling in the audio

def initAudio():
    src = aubio.source(VID_FILE_NAME)
    o = aubio.onset(method="specflux", buf_size=src.hop_size * 2,
                    hop_size=src.hop_size, samplerate=src.samplerate)
    onsets = []
    while True:
        samples, read = src()
        if o(samples):
            onsets.append(o.get_last())
        if read < src.hop_size:
            break
    return onsets, src.samplerate

# onsetPresent:
#
def onsetPresent(count, offsets, ocurr, fr, sr):
    curSample = count * sr / fr
    if offsets[ocurr] - curSample < THRESH_ONSET * sr:
        return True
    if ocurr > 0 and curSample - offsets[ocurr - 1] < THRESH_ONSET * sr:
        return True
    return False

# main():
# The main method in the program. First, gets the lowest key in the keyboard, from the user.
# Then, runs the algorithm for visual and audio transcription:
# 1) Initializes the keyboard
#       a) Crops the keyboard out of the piano
#       b) Generates a baseline image for comparison
#       c) Finds black keys using findContour(), and divides keys using these contours
# 2) Detects key presses
#       a) Loops through each frame
#       b) Calculates positive and negative image difference
#       c) Checks difference at each key to determine if key is pressed
# 3) Compare key presses to onsets detected
#       a) Find a list of all onsets in the audio
#       b) Compare each key press to a range of time in the onset list
#       c) Cancel key press if there is no nearby onset.
def main():
    lowest_key = -2
    while lowest_key < -1 or lowest_key > 52:
        lowest_key = int(
            input("What is the lowest WHITE key displayed on the keyboard? Enter 0 for A0, 1 for B0, 2 for C1 etc.\n"
                  "Note: check https://newt.phys.unsw.edu.au/jw/notes.html for reference\n"))
    #assume 52 keys, starting at a
    #get cropped keyboard from video
    bottomPiano, topPiano = cropPianoVert()

    #get baseline image (no hands) for comparison
    baseline_img = baseline(bottomPiano, topPiano) #grayscale
    height, width = baseline_img.shape[:2]
    leftPiano, rightPiano = cropPianoSide(baseline_img)
    baseline_img = baseline_img[0:height, leftPiano:(width - rightPiano)]
    height, width = baseline_img.shape[:2]

    # contour detection: figure out where keys are
    contourImg = baseline_img[int(height / 10):height, 0:width]
    baselineBlurred = cv2.GaussianBlur(contourImg, (1, 1), cv2.BORDER_DEFAULT)
    thresh = cv2.threshold(baselineBlurred, 120, 255, cv2.THRESH_BINARY)[1]
    threshInv = cv2.bitwise_not(thresh)

    baseline_img = cv2.GaussianBlur(baseline_img, (3, 3), cv2.BORDER_DEFAULT)

    # find contours
    contours, hierarchy = cv2.findContours(threshInv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # sum up total area in contours (ideally, total black key area)
    sumArea = 0
    for c in contours:
        M = cv2.moments(c)
        sumArea = sumArea + M["m00"]

    # loop over the contours (some code adapted from
    # https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/)
    a = 0
    keyEdges = [width]
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)

        if M["m00"] > sumArea / 80 and M["m00"] < sumArea/28:  # keys must be a certain area (avoid small noise)
            a = a + 1
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            # cv2.drawContours(baseline_img, [c], -1, (0, 255, 0), 2)
            # baseline_img = cv2.circle(baseline_img, (cX, cY), 7, (255, 255, 255), -1)
            # cv2.putText(baseline_img, "center", (cX - 20, cY - 20),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            keyEdges.append(cX)
            print(cX)
    keyEdges.append(0)
    diffEdges = np.diff(keyEdges)
    avg = np.average(diffEdges)

    print("avg", avg)
    ins = 0;
    for i in range(0, len(diffEdges)):
        print("diff", i, diffEdges[i])
        if math.fabs(diffEdges[i]) > math.fabs(avg):
            keyEdges.insert(i + ins + 1, (keyEdges[i + ins] + keyEdges[i + ins + 1]) / 2)
            ins += 1

    vid = cv2.VideoCapture(VID_FILE_NAME)
    fr = vid.get(cv2.CAP_PROP_FPS)
    count = 0

    mp = mido.open_output(MIDI_BUS_NAME)
    prevNotes = []
    onsets, sr = initAudio()
    ocurr = 0

    while True:
        count += 1
        frameStart = time.time()
        print("*****FRAME*****", count)
        #get frame
        _, frame = vid.read()
        if frame is None:
            break
        height, width = frame.shape[:2]

        #crop frame to only include piano, convert to grayscale
        pianoCropped = frame[int(topPiano):int(bottomPiano), leftPiano:(width - rightPiano)].copy()
        height, width = pianoCropped.shape[:2]
        current_gray = cv2.cvtColor(pianoCropped, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (3, 3), cv2.BORDER_DEFAULT)

        #images storing positive and negative differences
        diffImagePos = baseline_img.copy()
        diffImageNeg = baseline_img.copy()

        #compute difference between baseline and current grayscale images
        for i in range(current_gray.shape[0]):
            for j in range(current_gray.shape[1]):
                curr_pixel = current_gray.item(i, j)
                base_pixel = baseline_img.item(i, j)
                if curr_pixel > base_pixel:
                    diffImagePos.itemset((i,j), curr_pixel - base_pixel)
                    diffImageNeg.itemset((i,j), 0)
                else:
                    diffImagePos.itemset((i,j), 0)
                    diffImageNeg.itemset((i,j), base_pixel - curr_pixel)

        ret, threshDiffPos = cv2.threshold(diffImagePos, THRESH_DIFF_PRESSED, 255, cv2.THRESH_BINARY)
        ret, threshDiffNeg = cv2.threshold(diffImageNeg, THRESH_DIFF_PRESSED, 255, cv2.THRESH_BINARY)

        lastKeyPos = 0;
        curNotes = []

        for i in range(len(keyEdges) - 1, 0, -1):
            pianoCropped = cv2.line(pianoCropped, (int(keyEdges[i]), 0), (int(keyEdges[i]), height), (0, 255, 0), 1)

            thisKeyNeg = threshDiffNeg[0:int(height/8), int(keyEdges[i]):int(keyEdges[i-1])]
            thisKeyPos = threshDiffPos[0:int(height/8), int(keyEdges[i]):int(keyEdges[i-1])]
            # cv2.imshow('im', thisKeyNeg)
            # neg image: detect white key presses
            if np.mean(thisKeyNeg) > THRESH_MEAN_KEY: # note: do we need a different (larger) threshold for white
                                                      # keys between black keys? (bigger negative increase)
                print("neg", i)
                thisNote = convertMIDI(lowest_key, len(keyEdges) - 1 - i)
                curNotes.append(thisNote)

            if np.mean(thisKeyPos) > THRESH_MEAN_KEY:
                if lastKeyPos == 1:
                    print("black key pressed", i-1, i)
                    thatNote = convertMIDI(lowest_key, len(keyEdges) - 1 - i) - 1
                    curNotes.append(thatNote)
                lastKeyPos = 1;
                print("pos", i)
            else:
                lastKeyPos = 0;


        if len(curNotes) <= 15:
            if onsetPresent(count, onsets, ocurr, fr, sr):
                for curNote in curNotes:
                    if curNote not in prevNotes:
                        msg = mido.Message('note_on', note=curNote)
                        mp.send(msg)
            for curNote in prevNotes:
                if curNote not in curNotes:
                    msg = mido.Message('note_off', note=curNote)
                    mp.send(msg)
            prevNotes = curNotes
        if (count * sr / fr > onsets[ocurr]) and (ocurr < len(onsets) - 1):
            ocurr += 1

        # Display the various processed images for debugging or demo purposes
        # cv2.drawContours(pianoCropped, contours, -1, (0, 255, 0), 1)
        cv2.imshow('piano', pianoCropped)
        cv2.imshow('app', frame)
        cv2.imshow('baseline', baseline_img)
        cv2.imshow('diff POS', threshDiffPos)
        cv2.imshow('diff NEG', threshDiffNeg)
        cv2.imshow('grayscale piano', current_gray)
        # cv2.imshow('labeled', labels, cmap='gray')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frameEnd = time.time()
        timeDiff = frameEnd - frameStart
        if 1 / P_FPS > timeDiff:
            time.sleep(1 / P_FPS - timeDiff)
        else:
            print("FPS Limit Insufficient")

    vid.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    warnings.filterwarnings("ignore")
    main()