# import packages
import cv2
from rembg import remove
import easygui
from PIL import Image

frame = cv2.imread('Images/girl3.jpg')


# convert to grayscale of eachframes
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    faces = frame[y:y + h, x:x + w]
    cv2.imshow("face", faces)
    cv2.imwrite('Outputs/Detected_face.jpg', faces)

# Display the output
cv2.imshow('img', frame)

inputPath = easygui.fileopenbox(title='Select Image File')
outPutPath = easygui.filesavebox(title='Save File To...')

input = Image.open(inputPath)
outPut = remove(input)
outPut.save(outPutPath)
cv2.waitKey()
cv2.destroyAllWindows()
