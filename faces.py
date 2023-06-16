import cv2 as cv
from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="Minha-chave")
project = rf.workspace().project("face-detection-mik1i")
model = project.version(21).model

input_video = cv.VideoCapture('./assets/arsene.mp4')

if not input_video.isOpened():
    print("Error opening video file")
    exit(1)
    
width  = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

output_video = cv.VideoWriter( './saida/out.avi',cv.VideoWriter_fourcc(*'DIVX'), 24, (width, height))

while True:
    ret, frame = input_video.read()

    if not ret:
        break

    result = model.predict(frame, confidence=0.6).json()

    cv.rectangle(
            img=frame,
            pt1=(int(result['predictions'][0]['x']), int(result['predictions'][0]['y'])),
            pt2=(int(result['predictions'][0]['x']) + int(result['predictions'][0]['height']), int(result['predictions'][0]['y']) + int(result['predictions'][0]['width'])),
            color=(0,0,255),
            thickness=5
        )
    
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
output_video.release()
input_video.release()
cv.destroyAllWindows()