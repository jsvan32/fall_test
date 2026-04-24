import cv2
import numpy as np
from inference import get_roboflow_model

# ======================
# 配置
# ======================
API_KEY = "HRUKAM4JAN5Aha0u4Edr"
MODEL_ID = "fall-detection-mbldh/1"
CONF_THRESHOLD = 0.1

INPUT_VIDEO = "slipsv2.mp4"
OUTPUT_VIDEO = "fall_output_local_warning.mp4"

# 绘制参数
BOX_THICKNESS = 4               # 边界框粗细（原为2）
WARNING_DURATION_FRAMES = 5     # 警告持续帧数（用于平滑显示，本示例中每帧实时判断，此参数备用）

# ======================
# 加载模型
# ======================
print("正在加载模型...")
model = get_roboflow_model(model_id=MODEL_ID, api_key=API_KEY)
print("模型加载完成（本地推理模式）")

def draw_predictions(frame, predictions):
    """
    将检测到的所有目标统一视为摔倒 (fall)，并绘制加粗框 + 警告横幅。
    """
    fall_detected = False
    if not predictions:
        return frame, fall_detected

    h, w = frame.shape[:2]

    for pred in predictions:
        conf = pred.confidence
        if conf < CONF_THRESHOLD:
            continue

        # ---------- 关键修改：忽略模型输出的类别名，强制使用 "fall" ----------
        class_name = "fall"          # 强制改为 "fall"
        fall_detected = True         # 只要有任意检测框，就触发警告

        # 坐标转换
        x1 = int(pred.x - pred.width / 2)
        y1 = int(pred.y - pred.height / 2)
        x2 = int(pred.x + pred.width / 2)
        y2 = int(pred.y + pred.height / 2)

        # 绘制加粗边界框（红色）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), BOX_THICKNESS)

        # 标签文字
        label = f"fall {conf:.2f}"    # 标签也统一显示为 fall
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

    # 如果检测到摔倒（即任意有效检测），显示顶部警告横幅
    if fall_detected:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        warning_text = "⚠ FALL DETECTED! ⚠"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 55
        cv2.putText(frame, warning_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    1.2, (255, 255, 255), 3, cv2.LINE_AA)

    return frame, fall_detected

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {INPUT_VIDEO}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_count = 0
    print(f"开始处理视频（共 {total_frames} 帧）...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 本地推理
        results = model.infer(frame)[0]
        annotated, fall_flag = draw_predictions(frame.copy(), results.predictions)

        # （可选）控制台输出警告
        if fall_flag:
            print(f"⚠️ 第 {frame_count} 帧检测到摔倒！")

        out.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")

    cap.release()
    out.release()
    print(f"完成！输出视频已保存至：{OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()