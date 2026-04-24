import cv2
import numpy as np
from collections import deque
from inference import get_roboflow_model

# ======================
# 配置
# ======================
API_KEY = "HRUKAM4JAN5Aha0u4Edr"
MODEL_ID = "fall-detection-mbldh/1"
CONF_THRESHOLD = 0.15          # 置信度阈值
CAMERA_ID = 0
FRAME_SKIP = 2                 # 每隔N帧做一次推理，提升流畅度
BOX_THICKNESS = 3
SAVE_VIDEO = False
OUTPUT_VIDEO = "fall_realtime_output.mp4"

# 时序平滑：连续 TRIGGER_FRAMES 帧检测到摔倒才触发报警，减少误报
TRIGGER_FRAMES = 3
# 报警解除：连续 CLEAR_FRAMES 帧无摔倒才解除，避免闪烁
CLEAR_FRAMES = 5
# 历史窗口长度
HISTORY_LEN = max(TRIGGER_FRAMES, CLEAR_FRAMES) + 2

# 宽高比辅助：bbox 宽/高 > 此值时倾向判定为摔倒（人横躺）
FALL_ASPECT_RATIO = 1.2

# Roboflow 数据集标签反转映射
LABEL_MAP = {
    "fall": "stand",   # 模型输出 fall → 实际是 stand
    "stand": "fall",   # 模型输出 stand → 实际是 fall
}

# ======================
# 加载模型
# ======================
print("正在加载模型...")
model = get_roboflow_model(model_id=MODEL_ID, api_key=API_KEY)
print("模型加载完成，启动摄像头...")


def correct_label(raw_class: str) -> str:
    """修正 Roboflow 数据集的标签反转问题。"""
    return LABEL_MAP.get(raw_class.lower(), raw_class.lower())


def is_fall_by_aspect(w: float, h: float) -> bool:
    """宽高比辅助判断：横躺时宽 > 高。"""
    return (w / h) > FALL_ASPECT_RATIO if h > 0 else False


def draw_predictions(frame, predictions, fall_history: deque):
    """
    绘制检测结果，修正标签，结合时序平滑判断是否触发摔倒报警。
    返回 (annotated_frame, is_alarm)
    """
    w_frame = frame.shape[1]
    current_fall = False
    person_boxes = []  # (corrected_label, conf, x1, y1, x2, y2, raw_w, raw_h)

    if predictions:
        for pred in predictions:
            conf = pred.confidence
            if conf < CONF_THRESHOLD:
                continue

            raw_label = pred.class_name.lower()
            corrected = correct_label(raw_label)

            x1 = int(pred.x - pred.width / 2)
            y1 = int(pred.y - pred.height / 2)
            x2 = int(pred.x + pred.width / 2)
            y2 = int(pred.y + pred.height / 2)
            person_boxes.append((corrected, conf, x1, y1, x2, y2, pred.width, pred.height))

    # 判断当前帧是否有摔倒
    for corrected, conf, x1, y1, x2, y2, bw, bh in person_boxes:
        if corrected == "fall":
            # 模型说 fall + 宽高比也支持 → 高置信摔倒
            current_fall = True
            break
        if corrected == "stand" and is_fall_by_aspect(bw, bh):
            # 模型说 stand 但宽高比像横躺 → 也标记为摔倒（降低漏检）
            current_fall = True
            break

    # 更新时序历史
    fall_history.append(current_fall)

    # 时序判断：是否触发/解除报警
    recent = list(fall_history)
    alarm = False
    if sum(recent[-TRIGGER_FRAMES:]) >= TRIGGER_FRAMES:
        alarm = True
    # 已在报警状态时，需要连续 CLEAR_FRAMES 帧无摔倒才解除
    elif len(recent) >= CLEAR_FRAMES and sum(recent[-CLEAR_FRAMES:]) == 0:
        alarm = False

    # ---- 绘制 ----
    for corrected, conf, x1, y1, x2, y2, bw, bh in person_boxes:
        # 绘制颜色：stand=绿, fall=红
        is_fall_box = (corrected == "fall") or (corrected == "stand" and is_fall_by_aspect(bw, bh))
        color = (0, 0, 255) if is_fall_box else (0, 200, 0)
        display_label = "FALL" if is_fall_box else "STAND"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        label_text = f"{display_label} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 5), (x1 + lw, y1), color, -1)
        cv2.putText(frame, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 顶部报警横幅
    if alarm:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, 80), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        warning_text = "!! FALL DETECTED !!"
        ts = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        tx = (w_frame - ts[0]) // 2
        cv2.putText(frame, warning_text, (tx, 55), cv2.FONT_HERSHEY_DUPLEX,
                    1.2, (255, 255, 255), 3, cv2.LINE_AA)

    return frame, alarm


def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"分辨率: {width}x{height}, FPS: {fps}")

    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc,
                              int(fps) if fps > 0 else 20, (width, height))

    frame_count = 0
    last_predictions = None
    fall_history = deque(maxlen=HISTORY_LEN)
    alarm_active = False

    print("实时检测中... 按 'q' 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 FRAME_SKIP 帧推理一次
        if frame_count % FRAME_SKIP == 0:
            small_frame = cv2.resize(frame, (640, 480))
            results = model.infer(small_frame)[0]
            last_predictions = results.predictions

        annotated, alarm = draw_predictions(frame.copy(), last_predictions, fall_history)

        # 报警状态变化时打印
        if alarm and not alarm_active:
            print(f"[!] 第 {frame_count} 帧触发摔倒报警！")
            alarm_active = True
        elif not alarm and alarm_active:
            print(f"[√] 第 {frame_count} 帧摔倒报警解除")
            alarm_active = False

        # 左下角状态
        status_text = "ALARM" if alarm_active else "NORMAL"
        status_color = (0, 0, 255) if alarm_active else (0, 200, 0)
        cv2.putText(annotated, status_text, (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("Fall Detection - Real Time", annotated)

        if out:
            out.write(annotated)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("检测结束")


if __name__ == "__main__":
    main()
