# 脑卒中跌倒预防与安全管理 — 健康宣教与智能检测

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)](https://developer.mozilla.org/zh-CN/docs/Web/HTML)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)](https://developer.mozilla.org/zh-CN/docs/Web/CSS)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Roboflow-8A2BE2?style=flat)](https://roboflow.com/)

本项目包含两个互补模块：
1. 📱 **患者健康宣教平台**：纯前端网页，提供卒中后跌倒预防的结构化知识、风险自评与满意度调查。
2. 🎥 **跌倒智能检测系统**：基于YOLO的实时摄像头/视频跌倒检测，支持本地推理与报警。

旨在形成“教育-监测-预警”闭环，提升卒中患者安全。

## 🧠 项目背景

跌倒是脑卒中后常见的严重并发症。有效的预防需要**患者认知提升**与**环境智能监测**双管齐下。  
本仓库整合了：
- 一个**无需安装的Web应用**，将碎片化口头宣教转化为可重复访问的数字化教育资源；
- 一套**轻量级计算机视觉算法**，利用摄像头实时检测跌倒事件，适用于病房、居家护理场景。

## 📱 模块一：健康宣教与评估平台

### 功能亮点
- **结构化健康教育**：六大主题（风险因素、预防措施、家居改造、康复训练、应急处理），交互式呈现。
- **三套标准化评估问卷**（全部本地计算，保护隐私）：
  - 跌倒风险自评量表（10题，0-20分）
  - 健康知识掌握程度调查表（10维度，0-30分）
  - 患者满意度调查表（16条目，16-64分）
- **响应式设计**：适配手机、平板、电脑，支持屏幕阅读器无障碍访问。

### 快速使用
- **本地查看**：直接双击 `index.html` 用浏览器打开。
- **部署上线**：使用 Netlify 静态托管服务，无需服务器。


## 🎥 模块二：跌倒检测算法

基于 Roboflow 的 YOLO 模型（`fall-detection-mbldh/1`）实现跌倒检测，提供两种运行模式：

| 脚本文件 | 功能 | 输入源 | 输出 |
|-----------|------|--------|------|
| `fall_camera.py` | 实时摄像头检测 + 时序平滑报警 | 本地摄像头（ID 0） | 实时窗口显示，可选保存录像 |
| `fall_video.py` | 离线视频处理 | 视频文件（如 `slipsv2.mp4`） | 生成带标注的输出视频 |

### 算法特性
- 标签反向修正（模型原始标签 `fall` ↔ `stand` 互换，适配自有数据集）。
- 宽高比辅助判断（当人物检测框宽度大于高度时，强化为跌倒判定）。
- 时序平滑策略（连续多帧触发才报警，避免单帧误报闪烁）。
- 本地推理，无需联网，延迟低。

#### 1. 安装依赖
```bash
pip install -r requirements.txt