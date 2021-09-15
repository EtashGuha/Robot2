import sys
import cozmo
import datetime
import pickle
import time
import re
from datetime import datetime
from cozmo.util import degrees, distance_mm, speed_mmps
import numpy as np
from skimage import io, feature, filters, exposure, color
from sklearn import svm, metrics
from img_classification import ImageClassifier


def mission1(sdk_conn):
    '''
    Mission 1: Surveillance (Idle)

    Use your classifier to identify images from Cozmo's camera stream.
    If the image is "drone", "order", or "inspection", Cozmo should say the
    image name and the resulting mission is triggered. 
    Otherwise, the finite state machine stays in the Idle state.
    '''

    #######################
    # WRITE YOUR CODE HERE#
    #######################
    pass


def mission2(sdk_conn):
    '''
    Mission 2: Defuse the Bomb (activated by "order")

    |----|----|----|----|
    |    |    |    |    |
    A    B    C    D    E
    |    |    |    |    |
    |----|----|----|----|
    After a cube placed at location 'C', the Cozmo should start at location
    'D' while directly facing the cube, drive forward to pick up the cube,
    and drive forward to place the cube at location 'A'. Afterwards, the Cozmo
    will turn around to return to location 'D'.

    Return to the Idle state afterwards.
    '''

    #######################
    # WRITE YOUR CODE HERE#
    #######################
    pass


def mission3(sdk_conn):
    '''
    Mission 3: In the Heights (activated by "drone")

    Cozmo drives in an 'S' formation in the arena. During the mission, Cozmo
    an animation of your choice on the face (can be before, during, or after
    the formation).

    Return to the Idle state afterwards.
    '''

    #######################
    # WRITE YOUR CODE HERE#
    #######################
    pass


def mission4(sdk_conn):
    '''
    Mission 4: Burn Notice (activated by "inspection")

    Cozmo drives in an (approximately) 20-cm square formation. While driving, Cozmo
    should *slowly* raise and lower its lift repeatedly AND simultaneously say "I am
    not a spy". After finishing the square formation, Cozmo should have its lift lowered.

    Return to the Idle state afterwards.
    '''

    #######################
    # WRITE YOUR CODE HERE#
    #######################
    pass


def controller(sdk_conn):
    '''
    Create a finite state machine where the Cozmo stays in the Idle state (Mission 1)
    until Cozmo sees "drone", "order", or "inspection', resulting in a transition into
    another state (represented by Missions 2, 3, and 4). After resolving a non-Idle state,
    Cozmo returns back to the Idle state.

    HINT: If you're wondering why we may want the frames list, take a look at the attached
    calculations on Canvas.
    '''
    state = "none"
    frames = ["none"]

    while(True):

        #######################
        # WRITE YOUR CODE HERE#
        #######################
        pass


if __name__ == '__main__':
    cozmo.setup_basic_logging()

    # We are providing our own pickled ImageClassifier that you can use from Lab 2 and onward.
    # If you would like to use your own classifier, feel free to pickle your ImageClassifier
    # from Lab 1. More information on Python pickle: https://wiki.python.org/moin/UsingPickle
    with open('classifier_pickle', 'rb') as f:
        img_clf = pickle.load(f)

    try:
        cozmo.connect(controller)
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)