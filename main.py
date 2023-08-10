import sys
import cv2
import torch
import qtawesome
import mediapipe as mp
from modules.load_state import load_state
from models.with_mobilenet import PoseEstimationWithMobileNet
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QFileDialog
from demo.yolov8_demo import run_yolov8
from demo.lp_demo import run_demo


# 主界面
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.cap = cv2.VideoCapture(0)

    def init_ui(self):
        # 设置窗口标题、图标、大小
        self.setWindowTitle("All for Arknights")
        self.setWindowIcon(QIcon('demo/logo.png'))
        self.resize(960, 700)
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()
        self.left_widget.setLayout(self.left_layout)
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 1)
        self.main_layout.addWidget(self.right_widget, 0, 1, 12, 10)
        self.setCentralWidget(self.main_widget)

        self.left_close = QtWidgets.QPushButton("")
        self.left_close.clicked.connect(self.close)
        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit = QtWidgets.QPushButton("")
        self.left_visit.clicked.connect(self.showMinimized)
        self.left_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:green;}''')
        self.left_label_1 = QtWidgets.QPushButton("YOLO V8")
        self.left_label_1.setObjectName('left_label')
        self.left_label_2 = QtWidgets.QPushButton("Lighting pose")
        self.left_label_2.setObjectName('left_label')
        self.left_label_3 = QtWidgets.QPushButton("Media pipe")
        self.left_label_3.setObjectName('left_label')

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.play', color='white'), "摄像骨架")
        self.left_button_1.setObjectName('left_button')
        self.left_button_1.clicked.connect(self.yolov8_camera)
        self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.play', color='white'), "影像骨架")
        self.left_button_2.setObjectName('left_button')
        self.left_button_2.clicked.connect(self.yolov8_video)
        self.left_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.play', color='white'), "摄像骨架")
        self.left_button_3.setObjectName('left_button')
        self.left_button_3.clicked.connect(self.lighting_pose_camera)
        self.left_button_4 = QtWidgets.QPushButton(qtawesome.icon('fa.play', color='white'), "影像骨架")
        self.left_button_4.setObjectName('left_button')
        self.left_button_4.clicked.connect(self.lighting_pose_video)
        self.left_button_5 = QtWidgets.QPushButton(qtawesome.icon('fa.play', color='white'), "面部骨架")
        self.left_button_5.setObjectName('left_button')
        self.left_button_5.clicked.connect(self.media_pipe_face)
        self.left_button_6 = QtWidgets.QPushButton(qtawesome.icon('fa.play', color='white'), "手部骨架")
        self.left_button_6.setObjectName('left_button')
        self.left_button_6.clicked.connect(self.media_pipe_hand)
        self.left_button_7 = QtWidgets.QPushButton(qtawesome.icon('fa.play', color='white'), "躯干骨架")
        self.left_button_7.setObjectName('left_button')
        self.left_button_7.clicked.connect(self.media_pipe_body)
        self.left_button_8 = QtWidgets.QPushButton(qtawesome.icon('fa.play', color='white'), "全功能骨架")
        self.left_button_8.setObjectName('left_button')
        self.left_button_8.clicked.connect(self.media_pipe_full)

        self.left_layout.addWidget(self.left_visit, 0, 0, 1, 1)
        self.left_layout.addWidget(self.left_close, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_label_1, 1, 0, 1, 2)
        self.left_layout.addWidget(self.left_button_1, 2, 0, 1, 2)
        self.left_layout.addWidget(self.left_button_2, 3, 0, 1, 2)
        self.left_layout.addWidget(self.left_label_2, 4, 0, 1, 2)
        self.left_layout.addWidget(self.left_button_3, 5, 0, 1, 2)
        self.left_layout.addWidget(self.left_button_4, 6, 0, 1, 2)
        self.left_layout.addWidget(self.left_label_3, 7, 0, 1, 2)
        self.left_layout.addWidget(self.left_button_5, 8, 0, 1, 2)
        self.left_layout.addWidget(self.left_button_6, 9, 0, 1, 2)
        self.left_layout.addWidget(self.left_button_7, 10, 0, 1, 2)
        self.left_layout.addWidget(self.left_button_8, 11, 0, 1, 2)

        self.right_label1 = QLabel("")
        self.right_label1.setFixedSize(800, 600)

        self.right_playconsole_widget = QtWidgets.QWidget()  # 播放控制部件
        self.right_playconsole_layout = QtWidgets.QGridLayout()  # 播放控制部件网格布局层
        self.right_playconsole_widget.setLayout(self.right_playconsole_layout)
        self.console_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.pause', color='#F76677', font=18), "")
        self.console_button_1.setIconSize(QtCore.QSize(30, 30))
        self.console_button_1.clicked.connect(self.all_stop)
        self.right_playconsole_layout.addWidget(self.console_button_1, 0, 1)
        self.right_playconsole_layout.setAlignment(QtCore.Qt.AlignCenter)  # 设置布局内部件居中显示

        self.right_layout.addWidget(self.right_label1, 0, 1, 10, 10)
        self.right_layout.addWidget(self.right_playconsole_widget, 11, 1, 1, 10)
        self.right_layout.setAlignment(QtCore.Qt.AlignCenter)  # 设置布局内部件居中显示

        self.left_widget.setStyleSheet('''
            QPushButton{border:none;color:white;}
            QPushButton#left_label{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
            QWidget#left_widget{
                background:gray;
                border-top:1px solid white;
                border-bottom:1px solid white;
                border-left:1px solid white;
                border-top-left-radius:10px;
                border-bottom-left-radius: 10px;
            }
        ''')

        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QLabel#right_lable{
                border:none;
                font-size:16px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
        ''')

        self.right_playconsole_widget.setStyleSheet('''
            QPushButton{
                border:none;
            }
        ''')

        self.setWindowOpacity(0.9)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.main_layout.setSpacing(0)

    def all_stop(self):
        self.cap.release()
        self.right_label1.clear()

    def update_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将图像转换为QImage对象
        qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        # 将QImage对象转换为QPixmap对象
        pixmap = QPixmap.fromImage(qimage)
        # 将Pixmap对象设置给QLabel用于显示
        self.right_label1.setPixmap(pixmap)

    def yolov8_camera(self):
        self.right_label1.setFixedSize(800, 600)
        self.right_label1.setScaledContents(True)
        self.cap = cv2.VideoCapture(0)
        while True:
            # 读取一帧图像
            ret, frame = self.cap.read()
            if ret:
                # 调用run_demo函数处理图像并返回RGB图像数组信息
                rgb = run_yolov8(frame)
                # 更新图像显示
                self.update_image(rgb)
                # 按下q键退出循环
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 关闭摄像头
        self.cap.release()

    def yolov8_video(self):
        self.right_label1.setFixedSize(800, 600)
        self.right_label1.setScaledContents(True)
        # 打开文件选择对话框，选择视频文件
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            while True:
                # 读取一帧图像
                ret, frame = self.cap.read()
                if ret:
                    # 调用run_demo函数处理图像并返回RGB图像数组信息
                    rgb = run_yolov8(frame)
                    # 更新图像显示
                    self.update_image(rgb)
                else:
                    break
                # 按下q键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # 关闭视频文件
            self.cap.release()

    def lighting_pose_camera(self):
        self.right_label1.setFixedSize(800, 600)
        self.right_label1.setScaledContents(True)
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(
            r"./all_pt/checkpoint_iter_370000.pth",
            map_location='cpu')
        load_state(net, checkpoint)
        self.cap = cv2.VideoCapture(0)
        while True:
            # 读取一帧图像
            ret, frame = self.cap.read()
            if ret:
                # 调用run_demo函数处理图像并返回RGB图像数组信息
                rgb = run_demo(net, [frame], 256, False, 1, 1)
                # 更新图像显示
                self.update_image(rgb)
                # 按下q键退出循环
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 关闭摄像头
        self.cap.release()

    def lighting_pose_video(self):
        self.right_label1.setFixedSize(800, 600)
        self.right_label1.setScaledContents(True)
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(
            r"./all_pt/checkpoint_iter_370000.pth",
            map_location='cpu')
        load_state(net, checkpoint)
        # 打开文件选择对话框，选择视频文件
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            while True:
                # 读取一帧图像
                ret, frame = self.cap.read()
                if ret:
                    # 调用run_demo函数处理图像并返回RGB图像数组信息
                    rgb = run_demo(net, [frame], 256, False, 1, 1)
                    # 更新图像显示
                    self.update_image(rgb)
                else:
                    break
                # 按下q键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # 关闭视频文件
            self.cap.release()

    def media_pipe_face(self):
        self.right_label1.setFixedSize(800, 600)
        self.right_label1.setScaledContents(True)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic
        self.cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if success:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                    self.update_image(image)
                else:
                    break
                # cv2.imshow('MediaPipe Face', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        self.cap.release()

    def media_pipe_hand(self):
        self.right_label1.setFixedSize(800, 600)
        self.right_label1.setScaledContents(True)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic
        self.cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if success:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    self.update_image(image)
                else:
                    break
                # cv2.imshow('MediaPipe Face', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        self.cap.release()

    def media_pipe_body(self):
        self.right_label1.setFixedSize(800, 600)
        self.right_label1.setScaledContents(True)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic
        self.cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if success:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                            .get_default_pose_landmarks_style())
                    self.update_image(image)
                else:
                    break
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        self.cap.release()

    def media_pipe_full(self):
        self.right_label1.setFixedSize(800, 600)
        self.right_label1.setScaledContents(True)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic
        self.cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if success:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                            .get_default_pose_landmarks_style())
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    self.update_image(image)
                else:
                    break
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
