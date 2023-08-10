import cv2
from ultralytics import YOLO
import torch

def run_yolov8(img_bgr):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO('../all_pt/yolov8n-pose.pt')
    model.to(device)
    bbox_color = (150, 0, 0)  # 框的 BGR 颜色
    bbox_thickness = 2  # 框的线宽
    bbox_labelstr = {
        'font_size': 2,  # 字体大小
        'font_thickness': 4,  # 字体粗细
        'offset_x': 0,  # X 方向，文字偏移距离，向右为正
        'offset_y': -20,  # Y 方向，文字偏移距离，向下为正
    }
    kpt_color_map = {
        0: {'name': 'Nose', 'color': [0, 0, 255], 'radius': 6},  # 鼻尖
        1: {'name': 'Right Eye', 'color': [255, 0, 0], 'radius': 6},  # 右边眼睛
        2: {'name': 'Left Eye', 'color': [255, 0, 0], 'radius': 6},  # 左边眼睛
        3: {'name': 'Right Ear', 'color': [0, 255, 0], 'radius': 6},  # 右边耳朵
        4: {'name': 'Left Ear', 'color': [0, 255, 0], 'radius': 6},  # 左边耳朵
        5: {'name': 'Right Shoulder', 'color': [193, 182, 255], 'radius': 6},  # 右边肩膀
        6: {'name': 'Left Shoulder', 'color': [193, 182, 255], 'radius': 6},  # 左边肩膀
        7: {'name': 'Right Elbow', 'color': [16, 144, 247], 'radius': 6},  # 右侧胳膊肘
        8: {'name': 'Left Elbow', 'color': [16, 144, 247], 'radius': 6},  # 左侧胳膊肘
        9: {'name': 'Right Wrist', 'color': [1, 240, 255], 'radius': 6},  # 右侧手腕
        10: {'name': 'Left Wrist', 'color': [1, 240, 255], 'radius': 6},  # 左侧手腕
        11: {'name': 'Right Hip', 'color': [140, 47, 240], 'radius': 6},  # 右侧胯
        12: {'name': 'Left Hip', 'color': [140, 47, 240], 'radius': 6},  # 左侧胯
        13: {'name': 'Right Knee', 'color': [223, 155, 60], 'radius': 6},  # 右侧膝盖
        14: {'name': 'Left Knee', 'color': [223, 155, 60], 'radius': 6},  # 左侧膝盖
        15: {'name': 'Right Ankle', 'color': [139, 0, 0], 'radius': 6},  # 右侧脚踝
        16: {'name': 'Left Ankle', 'color': [139, 0, 0], 'radius': 6},  # 左侧脚踝
    }
    kpt_labelstr = {
        'font_size': 1.5,  # 字体大小
        'font_thickness': 3,  # 字体粗细
        'offset_x': 10,  # X 方向，文字偏移距离，向右为正
        'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
    }
    skeleton_map = [
        {'srt_kpt_id': 15, 'dst_kpt_id': 13, 'color': [0, 100, 255], 'thickness': 1},  # 右侧脚踝-右侧膝盖
        {'srt_kpt_id': 13, 'dst_kpt_id': 11, 'color': [0, 255, 0], 'thickness': 1},  # 右侧膝盖-右侧胯
        {'srt_kpt_id': 16, 'dst_kpt_id': 14, 'color': [255, 0, 0], 'thickness': 1},  # 左侧脚踝-左侧膝盖
        {'srt_kpt_id': 14, 'dst_kpt_id': 12, 'color': [0, 0, 255], 'thickness': 1},  # 左侧膝盖-左侧胯
        {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'color': [122, 160, 255], 'thickness': 1},  # 右侧胯-左侧胯
        {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'color': [139, 0, 139], 'thickness': 1},  # 右边肩膀-右侧胯
        {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [237, 149, 100], 'thickness': 1},  # 左边肩膀-左侧胯
        {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'color': [152, 251, 152], 'thickness': 1},  # 右边肩膀-左边肩膀
        {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'color': [148, 0, 69], 'thickness': 1},  # 右边肩膀-右侧胳膊肘
        {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [0, 75, 255], 'thickness': 1},  # 左边肩膀-左侧胳膊肘
        {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [56, 230, 25], 'thickness': 1},  # 右侧胳膊肘-右侧手腕
        {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [0, 240, 240], 'thickness': 1},  # 左侧胳膊肘-左侧手腕
        {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [224, 255, 255], 'thickness': 1},  # 右边眼睛-左边眼睛
        {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [47, 255, 173], 'thickness': 1},  # 鼻尖-左边眼睛
        {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [203, 192, 255], 'thickness': 1},  # 鼻尖-左边眼睛
        {'srt_kpt_id': 1, 'dst_kpt_id': 3, 'color': [196, 75, 255], 'thickness': 1},  # 右边眼睛-右边耳朵
        {'srt_kpt_id': 2, 'dst_kpt_id': 4, 'color': [86, 0, 25], 'thickness': 1},  # 左边眼睛-左边耳朵
        {'srt_kpt_id': 3, 'dst_kpt_id': 5, 'color': [255, 255, 0], 'thickness': 1},  # 右边耳朵-右边肩膀
        {'srt_kpt_id': 4, 'dst_kpt_id': 6, 'color': [255, 18, 200], 'thickness': 1}  # 左边耳朵-左边肩膀
    ]
    results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果
    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)
    # 预测框的 xyxy 坐标
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy()
    for idx in range(num_bbox):  # 遍历每个框
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]
        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[0]
        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)
        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                              bbox_labelstr['font_thickness'])
        bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度
        # 画该框的骨架连接
        for skeleton in skeleton_map:
            # 获取起始点坐标
            srt_kpt_id = skeleton['srt_kpt_id']
            srt_kpt_x = round(bbox_keypoints[srt_kpt_id][0])
            srt_kpt_y = round(bbox_keypoints[srt_kpt_id][1])
            srt_kpt_conf = bbox_keypoints[srt_kpt_id][2]  # 获取起始点置信度
            # print(srt_kpt_conf)
            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = round(bbox_keypoints[dst_kpt_id][0])
            dst_kpt_y = round(bbox_keypoints[dst_kpt_id][1])
            dst_kpt_conf = bbox_keypoints[dst_kpt_id][2]  # 获取终止点置信度
            # print(dst_kpt_conf)
            # 获取骨架连接颜色
            skeleton_color = skeleton['color']
            # 获取骨架连接线宽
            skeleton_thickness = skeleton['thickness']
            # 如果起始点和终止点的置信度都高于阈值，才画骨架连接
            if srt_kpt_conf > 0.5 and dst_kpt_conf > 0.5:
                # 画骨架连接
                img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                                   thickness=skeleton_thickness)
        # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = round(bbox_keypoints[kpt_id][0])
            kpt_y = round(bbox_keypoints[kpt_id][1])
            kpt_conf = bbox_keypoints[kpt_id][2]  # 获取该关键点置信度
            if kpt_conf > 0.5:
                # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
                img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
    return img_bgr