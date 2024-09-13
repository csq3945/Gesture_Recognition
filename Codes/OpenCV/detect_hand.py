import time
import numpy as np
import cv2 as cv
import queue
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarkerResult
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

WIDTH, HIGH = 1920, 1080
BUFFER_LEN = 20
ANGLE_THRESHOLD1 = 150  # 指跟关节
ANGLE_THRESHOLD2 = 150
ANGLE_THRESHOLD3 = 170
LINE_COLOR = (255, 0, 127)

GESTURE =  [[0, 0, 0, 0, 0],    # 前进
            [1, 1, 1, 1, 1],    # 后退
            [1, 1, 0, 0, 1],]    # 原地扭身
            # [0, 1, 1, 0, 0],    # 左平移
            # [0, 0, 1, 1, 1],    # 右平移
            # [1, 0, 0, 0, 1],    # 左旋转
            # [0, 1, 0, 1, 1]]    # 右旋转
COMMAND = {0:"前进", 1:"后退", 2:"原地扭身", 3:"左平移", 4:"右平移", 5:"左旋转", 6:"右旋转"}


def cv_imshow(name:str, image):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 800, 450)
    cv.imshow(name, image)

# 检查队列里所有元素是否相同               
def check_queue(q):
    if q.qsize() == q.maxsize:
        items = list(q.queue)
        if all(x == items[0] for x in items):
            return items[0]
    return 7

class HandDetect:
    def __init__(self):
        # 导入模型并配置检测，其他三个参数使用默认0.5
        hand_options = HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='.\\hand_landmarker.task'),
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.finish_hands_callback
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        
        # 检测结果
        self.hand_result = None
        # 关节角度，节点对应模型
        self.joints_angles = dict()
        # 五指状态，大拇指->小拇指，0->4，伸直为1，弯曲为0
        self.fingers_states = dict()
        # 识别手势
        self.classify_gesture_result = None
    
    # 放在主循环中运行
    def run(self, frame):
        frame_np_as_array = np.asarray(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame_np_as_array)
        self.hand_detector.detect_async(mp_image, int(time.time()*1000))
        
    # 绘制手部节点及连线
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

        return annotated_image

    # 向量归一化
    def normalize_vector(self, vector):
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            return vector
        return vector / magnitude
    
    # 计算一个关节（middle）角度
    def calculat_joint_angle(self, root_index, middle_index, end_index, hand_landmarks):
        # 获取连续的三个节点坐标信息
        root = hand_landmarks[root_index]
        middle = hand_landmarks[middle_index]
        end = hand_landmarks[end_index]
        # 获取接中间节点的两个向量
        vec1 = np.array([root.x - middle.x, root.y - middle.y, root.z - middle.z])
        vec2 = np.array([end.x - middle.x, end.y - middle.y, end.z - middle.z])
        # 向量归一化
        vec1_norm = self.normalize_vector(vec1)
        vec2_norm = self.normalize_vector(vec2)
        # 计算向量点积即为cos值
        dot_product_result = np.dot(vec1_norm, vec2_norm)
        angle = np.rad2deg(np.arccos(dot_product_result))
        return angle

    # 计算五指的三个关节角度
    def calculate_all_fingers_angles(self):
        self.joints_angles.clear()
        hand_landmarks = self.hand_result.hand_world_landmarks[0]
        # 拇指
        for i in range(3):
            self.joints_angles[1+i] = self.calculat_joint_angle(i, 1+i, 2+i, hand_landmarks)
        # 四指
        for i in range(4):
            self.joints_angles[5+i*4] = self.calculat_joint_angle(0,     5+i*4, 6+i*4, hand_landmarks)
            self.joints_angles[6+i*4] = self.calculat_joint_angle(5+i*4, 6+i*4, 7+i*4, hand_landmarks)
            self.joints_angles[7+i*4] = self.calculat_joint_angle(6+i*4, 7+i*4, 8+i*4, hand_landmarks)
        
    # 判断一根手指是否弯曲
    def detect_one_finger(self, angle1, angle2, angle3):
        counter = 0
        if angle1 > ANGLE_THRESHOLD1:
            counter += 1
        if angle2 > ANGLE_THRESHOLD2:
            counter += 1
        if angle3 > ANGLE_THRESHOLD3:
            counter += 1
        if counter > 1:
            return 1
        else:
            return 0

    # 判断五指的状态
    def detect_five_finger(self):
        self.fingers_states.clear()
        for i in range(5):
            self.fingers_states[i] = self.detect_one_finger(self.joints_angles[1+i*4], self.joints_angles[2+i*4], self.joints_angles[3+i*4])
    
    # 对手势进行分类
    def classify_gesture(self):
        for i in range(len(GESTURE)):
            k = 0
            for j in range(5):
                if GESTURE[i][j] == self.fingers_states[j]:
                    k += 1
            if k == 5:
                self.classify_gesture_result = i
                return
        self.classify_gesture_result = 7

    # 检测完成运行回调函数
    def finish_hands_callback(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.hand_result = result
        # print(f"result:\n{self.hand_result}")
        if len(self.hand_result.hand_world_landmarks) > 0:
            self.calculate_all_fingers_angles()
            # print(f"angles:\n{self.joints_angles}")
            self.detect_five_finger()
            # print(f"states:\n{self.fingers_states}")
            self.classify_gesture()
            
            # if self.classify_gesture_result != 7:
            #     # 打印输出最终结果
            #     print(f"classify_gesture_result: {COMMAND[self.classify_gesture_result]}")
           
if __name__ == "__main__":
    # 创建队列缓冲，对结果进行滤波
    result_buffer = queue.Queue(maxsize = BUFFER_LEN)

    # 计算帧率
    frame_count = 0
    fps = 0
    start_time = time.time()

    hand = HandDetect()

    cap = cv.VideoCapture(".\\Media\\video2.mp4")
    # cap.set(cv.CAP_PROP_FPS, 30)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:

        ret, frame = cap.read()
        if not ret:
            print(f"Can't receive frame (stream end?). Exiting ...")
            break
        
        hand.run(frame)

        # 轮廓绘制
        if hand.hand_result:
            frame = hand.draw_landmarks_on_image(frame, hand.hand_result)
            # 结果滤波
            if result_buffer.full():
                result_buffer.get()
            result_buffer.put(hand.classify_gesture_result)
            result = check_queue(result_buffer)
            # 打印输出最终结果
            if hand.classify_gesture_result < len(GESTURE):
                # 打印输出最终结果
                print(f"classify_gesture_result: {COMMAND[hand.classify_gesture_result]}")

        # 计算帧率
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1.0:
            fps = frame_count
            frame_count = 0
            start_time = current_time
        cv.putText(frame, f"FPS:{fps}", (5, 30), cv.FONT_HERSHEY_SIMPLEX, 1, LINE_COLOR, 2, cv.LINE_AA)

        cv.imshow("frame", frame)
        if cv.waitKey(10) == 27:
            break
    cap.release()
    cv.destroyAllWindows()








