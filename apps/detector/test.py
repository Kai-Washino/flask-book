from pathlib import Path
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2

def detect_objects(input_image_path, output_image_path, threshold=0.5):
    # モデルのロード
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 入力画像の読み込みと前処理
    image = Image.open(input_image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)

    # 推論
    with torch.no_grad():
        output = model(input_tensor)

    # 結果の取得
    output = output[0]
    print(output)
    boxes = output["boxes"].numpy()
    labels = output["labels"].numpy()
    scores = output["scores"].numpy()

    # スコアのしきい値でフィルタリング
    indices = np.where(scores > threshold)[0]
    image_array = np.array(image)

    for idx in indices:
        box = boxes[idx]
        label = labels[idx]
        score = scores[idx]
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        cv2.rectangle(image_array, start_point, end_point, (255, 0, 0), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(image_array, text, (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 結果画像を保存
    output_image = Image.fromarray(image_array)
    output_image.save(output_image_path)    
    print(f"結果画像を保存しました: {output_image_path}")

if __name__ == '__main__':
    model_path = "C:\\Users\\S2\\Documents\\勉強\\Web\\FlaskによるWebアプリ開発入門\\flaskbook\\apps\\detector\\model.pt"
    input_image_path = 'C:\\Users\\S2\\Documents\\勉強\\Web\\FlaskによるWebアプリ開発入門\\flaskbook\\apps\\images\\images.jpg'
    output_image_path = 'C:\\Users\\S2\\Documents\\勉強\\Web\\FlaskによるWebアプリ開発入門\\flaskbook\\apps\\images\\output.jpg'
    detect_objects(input_image_path, output_image_path)
    # detect_objects(model_path, input_image_path, output_image_path)    