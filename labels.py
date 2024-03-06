
# Gán nhán Yolo
import os
import cv2
import numpy as np
import argparse
#Thêm các class name vào đây tương ứng
CLASSES = ["0"]

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_dir, output_dir):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_name)
            original_image = cv2.imread(image_path)
            [height, width, _] = original_image.shape

            length = max((height, width))
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = original_image

            scale = length / 640

            blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
            model.setInput(blob)
            outputs = model.forward()
            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []

            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                if maxScore >= 0.60:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2],
                        outputs[0][i][3],
                    ]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

            detections = []
            yolo_boxes = []

            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    "class_id": class_ids[index],
                    "class_name": CLASSES[class_ids[index]],
                    "confidence": scores[index],
                    "box": box,
                    "scale": scale,
                }
                detections.append(detection)

                # Convert bounding box to YOLO format

                # Draw bounding box on the image
                draw_bounding_box(
                    original_image,
                    class_ids[index],
                    scores[index],
                    round(box[0] * scale),
                    round(box[1] * scale),
                    round((box[0] + box[2]) * scale),
                    round((box[1] + box[3]) * scale),
                )
                yolo_box = [
                    CLASSES[class_ids[index]],  # class index
                    (box[0] * scale + (box[2] * scale) / 2) / width,  # center_x (normalize by width)
                    (box[1] * scale + (box[3] * scale) / 2) / height,  # center_y (normalize by height)
                    (box[2] * scale) / width,  # width (normalize by width)
                    (box[3] * scale) / height  # height (normalize by height)
                ]
                yolo_boxes.append(yolo_box)
            label_dir = os.path.join(output_dir, "labels")
            os.makedirs(label_dir, exist_ok=True)
            # Save YOLO labels to a file
            label_file_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))
            with open(label_file_path, "w") as label_file:
                for yolo_box in yolo_boxes:
                    label_file.write(
                        f"{yolo_box[0]} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}\n"
                    )
            output_dir = os.path.join(output_dir, "images")
            result_image_path = os.path.join(output_dir, image_name.replace(".jpg", "_p.jpg"))
            cv2.imwrite(result_image_path, original_image)
            print(f"Lưu anh: {result_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Thêm onnx vào
    parser.add_argument("--model", default="weights/best.onnx", help="Input your ONNX model.")
    #Thêm thư mục cần gán nhán vào
    parser.add_argument("--input_dir", default="", help="Directory containing input images.")
    parser.add_argument("--output_dir", default="results", help="Directory to save results.")
    args = parser.parse_args()
    main(args.model, args.input_dir, args.output_dir)
