import cv2
import numpy as np
import face_recognition



bezos_face = face_recognition.load_image_file("faces/Bezos.jpg")
bezos_face_encoding = face_recognition.face_encodings(bezos_face)[0]

gates_face = face_recognition.load_image_file("faces/Gates.jpg")
gates_face_encoding = face_recognition.face_encodings(gates_face)[0]

musk_face = face_recognition.load_image_file("faces/Musk.jpg")
musk_face_encoding = face_recognition.face_encodings(musk_face)[0]

chris_face = face_recognition.load_image_file("faces/Chris.jpg")
chris_face_encoding = face_recognition.face_encodings(chris_face)[0]

known_face_encodings = [
    bezos_face_encoding,
    gates_face_encoding,
    musk_face_encoding,
    chris_face_encoding
]


known_face_names = [
    "Jeff Bezos",
    "Bill Gates",
    "Elon Musk",
    "Chris"
]



image = cv2.imread("test_image/test Musk.jpg")


# Initialize some variables
face_names = []





# Resize image to 1/4 size for faster face recognition processing
small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb_small_image = small_image[:, :, ::-1]



# Find all the faces and face encodings in the current image
face_locations = face_recognition.face_locations(rgb_small_image)
face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)



# Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    # Draw a box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (255, 166, 128), 2)

    # Draw a label with a name below the face
    cv2.rectangle(image, (left, bottom-35), (right, bottom), (255, 166, 128), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1, (0, 0, 0), 1)

cv2.namedWindow("Course Work", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Course Work", 1400,800)
cv2.imshow("Course Work", image)
cv2.waitKey()