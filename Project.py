import cv2 
import mediapipe as mp

# Initialize mediapipe's drawing and pose detection tools 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set up the webcam capture (0 means the default webcam)
cap = cv2.VideoCapture(0)

# Use Mediapipe Pose model to detect and track body pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    # Initialize variables for counting pull-ups
    pull_ups = 0
    prev_y = 0  # Previous Y position of the nose
    state = "down"  # Start in the "down" position

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        # Convert the captured frame to RGB format (Mediapipe works with RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Define the position of the lines to count pull-ups (upper and lower limits)
        upper_line_start = (0, 100)  # Starting point of the upper line
        upper_line_end = (frame.shape[1], 100)  # Ending point of the upper line
        lower_line_start = (0, 350)  # Starting point of the lower line
        lower_line_end = (frame.shape[1], 350)  # Ending point of the lower line
        
        # Define the color and thickness for the lines
        line_color = (0, 255, 0)  # Green color for the lines
        line_thickness = 2  # Thickness of the lines
        
        # Draw the lines on the frame to visualize the pull-up range
        frame = cv2.line(frame, upper_line_start, upper_line_end, line_color, line_thickness)
        frame = cv2.line(frame, lower_line_start, lower_line_end, line_color, line_thickness)

        # Use Mediapipe Pose model to detect landmarks (like the nose, shoulders, etc.)
        results = pose.process(image)

        # If pose landmarks are found (like the nose), process the data
        if results.pose_landmarks:
            # Get the position of the nose landmark
            nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            # Calculate the Y position of the nose in pixel coordinates
            y = int(nose_landmark.y * frame.shape[0])
            
            # Draw a red circle around the nose for visual feedback
            cv2.circle(frame, (int(nose_landmark.x * frame.shape[1]), y), 5, (0, 0, 255), -1)

            # Display the Y coordinate of the nose at the top right corner of the frame
            cv2.putText(frame, f"y: {y}", (frame.shape[1] - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Check if the nose has crossed the lines to detect a pull-up
            if state == "down":  # If the person is in the down position
                if prev_y > lower_line_start[1] and y <= lower_line_start[1]:  # Nose crosses the lower line
                    state = "up"  # Change state to "up"
            elif state == "up":  # If the person is in the up position
                if prev_y < upper_line_start[1] and y >= upper_line_start[1]:  # Nose crosses the upper line
                    state = "down"  # Change state back to "down"
                    pull_ups += 1  # Increase the pull-up count
                
            # Update the previous Y position for the next frame
            prev_y = y
        
        # Display the current pull-up count on the frame
        cv2.putText(frame, f"Pull-ups: {pull_ups}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the updated frame with landmarks and pull-up count
        cv2.imshow('Nose detection', frame)

        # Exit the loop if the "q" key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
