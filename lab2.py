import sys
import cozmo
import datetime
import pickle
import time
import re
from datetime import datetime
from cozmo.util import degrees, distance_mm, speed_mmps
from checker_cozmo import RobotStateDisplay
import numpy as np
from skimage import io, feature, filters, exposure, color
from sklearn import svm, metrics
from imgclassification_sol import ImageClassifier

both_speed = .5
turn_speed = (.1, .5)

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
    robot = sdk_conn.wait_for_robot()
    robot.camera.enable_auto_exposure()
    robot.camera.image_stream_enabled = True
    latest_image = robot.world.latest_image
    
    next_state = "Idle"
    if latest_image is not None:
        print(np.array(latest_image.raw_image).shape)
        input_img = np.expand_dims(np.array(latest_image.raw_image), axis=0)
        extracted_data = img_clf.extract_image_features(input_img)

        label = img_clf.predict_labels(extracted_data)
        robot.say_text(label[0]).wait_for_completed()
        next_state = label[0] if label[0] != 'none' and label[0] != 'place' else "Idle"
    return next_state
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

    robot = sdk_conn.wait_for_robot()
    robot.move_lift(-1)
    time.sleep(1)
    # robot.drive_straight(distance_mm(1000), speed_mmps(1000)).wait_for_completed()
    robot.drive_straight(distance_mm(65.1), speed_mmps(1000)).wait_for_completed()
    robot.move_lift(1)
    time.sleep(1)
    # robot.drive_straight(distance_mm(1000), speed_mmps(1000)).wait_for_completed()
    robot.drive_straight(distance_mm(330.2), speed_mmps(1000)).wait_for_completed()
    robot.move_lift(-1)
    time.sleep(1)
    robot.drive_straight(distance_mm(-30), speed_mmps(1000)).wait_for_completed()
    robot.turn_in_place(degrees(180)).wait_for_completed()
    robot.drive_straight(distance_mm(300.2), speed_mmps(1000)).wait_for_completed()

    time.sleep(1)
    return "Idle"


def mission3(sdk_conn):
    '''
    Mission 3: In the Heights (activated by "drone")

    Cozmo drives in an 'S' formation in the arena. During the mission, Cozmo
    an animation of your choice on the face (can be before, during, or after
    the formation).

    Return to the Idle state afterwards.
    '''
    robot = sdk_conn.wait_for_robot()
    print("Mission 3")
    # robot.drive_wheels(80,40)
    # time.sleep(2)
    intial_time = time.time()
    curr_time = 0


    while curr_time < 6.283:
        # if time.time() - curr_time < 1:
        #     continue

        curr_time = time.time() - intial_time 
        left_wheel_speed = np.sin(curr_time + 3*np.pi/4) + 1
        right_wheel_speed = np.sin(curr_time + 1*np.pi/4) + 1
        print("LW: {} RW: {}".format(left_wheel_speed, right_wheel_speed))

        robot.drive_wheels(50 * left_wheel_speed, 50 * right_wheel_speed)

    robot.drive_wheels(0, 0)

    return "Idle"
        




    


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
    robot = sdk_conn.wait_for_robot()
    robot.drive_wheels(both_speed, both_speed, duration = 1).wait_for_completion()
    


def controller(sdk_conn):
    '''
    Create a finite state machine where the Cozmo stays in the Idle state (Mission 1)
    until Cozmo sees "drone", "order", or "inspection', resulting in a transition into
    another state (represented by Missions 2, 3, and 4). After resolving a non-Idle state,
    Cozmo returns back to the Idle state.

    HINT: If you're wondering why we may want the frames list, take a look at the attached
    calculations on Canvas.
    '''
    state = "Idle"
    frames = ["none"]

    while(True):
        if state == "Idle":
            state = mission1(sdk_conn)
        elif state == "order":
            state = mission2(sdk_conn)
        elif state == "drone":
            state = mission3(sdk_conn)


        


if __name__ == '__main__':
    cozmo.setup_basic_logging()

    # We are providing our own pickled ImageClassifier that you can use from Lab 2 and onward.
    # If you would like to use your own classifier, feel free to pickle your ImageClassifier
    # from Lab 1. More information on Python pickle: https://wiki.python.org/moin/UsingPickle
    with open('classifier_pickle', 'rb') as f:
        img_clf = pickle.load(f)

    try:
        sdk_conn = cozmo.connect(controller)
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)