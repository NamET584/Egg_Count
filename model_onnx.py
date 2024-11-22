from datetime import datetime
from typing import List
import numpy as np
import cv2
import onnxruntime as ort

class Yolov11_Onnx:
    def __init__(self, onnx_model_path: str, input_shape: tuple[int, int] = (640, 640), 
                 confidence_threshold: float = 0.3, nms_threshold: float = 0.85, 
                 label_list: List[str] = None):
        
        self.onnx_model_path = onnx_model_path
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.label_list = label_list if label_list else ["Chicken", "Egg"]
        self.session = ort.InferenceSession(self.onnx_model_path)

    def _preprocessing(self, frame):
        original_height, original_width = frame.shape[:2]
        self.resize_ratio_w = original_width / self.input_shape[0]
        self.resize_ratio_h = original_height / self.input_shape[1]
        
        input_img = cv2.resize(frame, self.input_shape)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.ascontiguousarray(input_img)

        input_img = input_img / 255.0
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def _postprocessing(self, output, image_shape):
        output = np.array(output)
        x_center = output[0,0,0, :]
        y_center = output[0,0,1, :]
        w = output[0,0,2, :]
        h = output[0,0,3, :]
        confidence = output[0, 0, 4:, :]

        class_id = np.argmax(confidence, axis=0)
        max_class_prob = np.max(confidence, axis=0)
        print(max_class_prob)
        print(output.shape)
        # Filter detections based on confidence threshold
        mask = (max_class_prob > self.confidence_threshold)
        print(mask)
        detections = [
            [
                x_center[i] * self.resize_ratio_w,
                y_center[i] * self.resize_ratio_h,
                w[i] * self.resize_ratio_w,
                h[i] * self.resize_ratio_h,
                self.label_list[class_id[i]] if self.label_list else str(class_id[i]),
                max_class_prob[i]
            ]
            for i in range(len(mask)) if mask[i]
        ]
        print(len(detections))
        if detections:
            boxes = np.array([[d[0] - d[2] / 2, d[1] - d[3] / 2, d[0] + d[2] / 2, d[1] + d[3] / 2] for d in detections])
            confidences = np.array([d[5] for d in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.confidence_threshold, self.nms_threshold)
            if len(indices) > 0:
                detections = [detections[i] for i in indices]
        print(detections)
        return detections

    def drawbox(self, frame, detections):
        num_eggs = 0
        num_chickens = 0

        for detection in detections:
            x_center, y_center, w, h, lbl, conf = detection
            x = x_center - w / 2
            y = y_center - h / 2
            x_max = x_center + w / 2
            y_max = y_center + h / 2
            class_name = lbl

            # Đếm số lượng trứng và gà
            if class_name == "Egg":
                num_eggs += 1
            elif class_name == "Chicken":
                num_chickens += 1

            # Vẽ bounding box cho từng đối tượng
            cv2.rectangle(frame, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 1)
            cv2.putText(frame, class_name, (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255],
                        thickness=2)
        
        # Chèn số lượng trứng và gà lên ảnh
        cv2.putText(frame, f"Eggs: {num_eggs}, Chickens: {num_chickens}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

def main():
    # Đường dẫn ảnh và file ONNX
    image_path = "D:/Egg_Chicken_Count/image/EggCheck_image_2016-06-18_1030.jpg"
    model_path = "D:/Egg_Chicken_Count/best.onnx"

    # Đọc ảnh
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    # Khởi tạo lớp Yolov11_Onnx
    yolov11 = Yolov11_Onnx(model_path)

    # Tiền xử lý ảnh
    input_tensor = yolov11._preprocessing(frame)

    # Chạy mô hình ONNX
    input_name = yolov11.session.get_inputs()[0].name
    output = yolov11.session.run(None, {input_name: input_tensor})

    # Hậu xử lý
    detections = yolov11._postprocessing(output, frame.shape)

    # Vẽ bounding box lên ảnh
    result_frame = yolov11.drawbox(frame, detections)

    # Hiển thị ảnh kết quả
    cv2.imshow("Detection Result", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
