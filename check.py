import cv2

def draw_boxes(image_path, yolo_labels):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    for label in yolo_labels:
        class_id, center_x, center_y, bbox_width, bbox_height = map(float, label.split())
        center_x = int(center_x * width)
        center_y = int(center_y * height)
        bbox_width = int(bbox_width * width)
        bbox_height = int(bbox_height * height)

        x = center_x - bbox_width // 2
        y = center_y - bbox_height // 2
        x_plus_w = x + bbox_width
        y_plus_h = y + bbox_height

        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Thêm ảnh kiểm tra vào đây
image_path = ""
#Thêm nhãn Yolo vào đây
yolo_labels = ["0 0.486334 0.538187 0.700302 0.615769"]
draw_boxes(image_path, yolo_labels)


