import cv2

# Load pre-trained face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load an image
image = cv2.imread("notPeople.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1, 
    minNeighbors=5, 
    minSize=(200, 200)
    )

print(f"Found {len(faces)} faces!")

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Resize to fixed width while keeping aspect ratio
desired_width = 800
height = int(image.shape[0] * (desired_width / image.shape[1]))
resized_image = cv2.resize(image, (desired_width, height))

# Show the result
cv2.imshow("Faces Detected", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

