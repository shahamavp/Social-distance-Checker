import cv2
import numpy as np

# Placeholder function for gender prediction (replace this with your actual gender classification model)
def predict_gender(face_roi):
    # Placeholder: Simulate gender prediction using a simple check (change this)
    # Replace this logic with your actual gender classification model prediction
    # For demonstration purposes, this example assumes it predicts gender based on the face's width
    face_width = face_roi.shape[1]
    if face_width > 100:
        return "Male"
    else:
        return "Female"

# Initialize OpenCV's HOG people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam
video = cv2.VideoCapture("C:\DL projects\Social Distancing Checker\pexels-george-morina-5330839 (1080p).mp4")

while True:
    success, frame = video.read()

    # Detect people in the frame
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)

    if len(boxes) > 0:
        # Loop through detected people
        for x, y, w, h in boxes:
            # Draw rectangles around detected people
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the detected face for gender classification (assuming face detected)
            face_roi = frame[y:y + h, x:x + w]

            # Perform gender classification on the cropped face
            gender = predict_gender(face_roi)  # Replace with your actual gender prediction code

            # Display gender label on the frame
            cv2.putText(frame, f'Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # Calculate distance between detected people (for demonstration purposes)
#         # You'll need a more sophisticated approach for accurate distance estimation
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                x1, y1, w1, h1 = boxes[i]
                x2, y2, w2, h2 = boxes[j]

#                 # Calculate Euclidean distance between centers of detected people
                distance = np.sqrt(((x2 + w2/2) - (x1 + w1/2)) ** 2 + ((y2 + h2/2) - (y1 + h1/2)) ** 2)

#                 # Convert pixel distance to feet (approximate, based on your scene)
                distance_feet = distance * 0.3  # Adjust this conversion factor based on your camera and scene

#                 # Display distance between people
                cv2.putText(frame, f'Distance: {distance_feet:.2f} feet', (x1, y1 + h1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    # Display the frame
    cv2.imshow('Social Distancing Checker', frame)
    cv2.imshow('Gender Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video.release()
cv2.destroyAllWindows()