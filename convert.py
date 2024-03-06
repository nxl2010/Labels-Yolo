#Chuyển Yolo file pt sang ONNX

from ultralytics import YOLO
#Thêm file yolo.pt vào đây
model = YOLO("weight/best.pt")

path = model.export(format="onnx")