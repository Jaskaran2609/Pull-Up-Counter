import cv2
import mediapipe as mp
k = int(input("enter the number"))
li = [0,3,4,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11,10,10,9,12,11]
# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# k = int("input the number: ")
# Initialize video capture device
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    pull_ups = 0
    prev_y = 0
    count = 0
    state = "down"  # Initial state
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Define start and end points for the pull-up counting lines
        upper_line_start = (0, 100)
        upper_line_end = (frame.shape[1], 100)
        lower_line_start = (0, 350)
        lower_line_end = (frame.shape[1], 350)
        
        # Define line color and thickness
        line_color = (0, 255, 0)
        line_thickness = 2
        
        # Draw the pull-up counting lines on the frame
        frame = cv2.line(frame, upper_line_start, upper_line_end, line_color, line_thickness)
        frame = cv2.line(frame, lower_line_start, lower_line_end, line_color, line_thickness)

        # Set the image as input to the pose model
        results = pose.process(image)

        # Extract the nose landmark
        if results.pose_landmarks:
            nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            # Get the y coordinate of the nose landmark
            y = int(nose_landmark.y * frame.shape[0])
            
            # Draw a red circle around the nose landmark
            cv2.circle(frame, (int(nose_landmark.x * frame.shape[1]), y), 5, (0, 0, 255), -1)

            # Display the y coordinate of the nose position at the top right corner
            cv2.putText(frame, f"y: {y}", (frame.shape[1] - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Check if the nose landmark has crossed the pull-up counting lines
            if state == "down":
                if prev_y > lower_line_start[1] and y <= lower_line_start[1]:
                    state = "up"
            elif state == "up":
                if prev_y < upper_line_start[1] and y >= upper_line_start[1]:
                    state = "down"
                    pull_ups += 1
                
            # Update the previous y coordinate
            prev_y = y
        
        # Display the pull-up count on the frame
        cv2.putText(frame, f"Pull-ups: {pull_ups}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Nose detection', frame)
        # if pull_ups==li[k]:
        #     # cv2.putText(frame, f"good job you are done", (10, 50),
        #     #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     break
        # Exit loop on "q" key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        
        

# Release the capture
cap.release()
cv2.destroyAllWindows()
if pull_ups==li[k]:
    img=cv2.imread("1.jpg",1)
    cv2.imshow("is the answer correct",img)
    cv2.waitKey(0)
else:
    img=cv2.imread("0.jpg",1)
    cv2.imshow("uh oh wrong answer correct",img)
    cv2.waitKey(0)