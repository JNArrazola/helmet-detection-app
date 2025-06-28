# Source: https://learnopencv.com/train-yolov8-on-custom-dataset/

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
   data='Helmet.v1i.yolov8/data.yaml',
   imgsz=640,
   epochs=50 ,
   batch=8,
   name='yolov8n_custom')