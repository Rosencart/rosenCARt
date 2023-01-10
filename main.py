import cv2

# Load the pretrained model
model = cv2.dnn.readNetFromTensorflow("model.pb")

# Get the names of the output layers
output_layers = model.getUnconnectedOutLayersNames()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    _, frame = cap.read()

    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)

    # Set the input to the model
    model.setInput(blob)

    # Run the forward pass through the network
    outs = model.forward(output_layers)

    # Get the class with the highest confidence
    class_id = int(outs[0][0][0][1])
    confidence = outs[0][0][0][2]

    # Draw the class label and confidence on the frame
    label = "Class: " + str(class_id)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    label = "Confidence: " + str(confidence)
    cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Traffic Sign Recognition", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
