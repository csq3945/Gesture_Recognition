import cv2 as cv
import numpy as np
import queue
import time

WIDTH, HIGH = 1920, 1080
LINE_COLOR = (255, 0, 127)

BUFFER_LEN = 5

# 圆形轮廓
CIRCLE_AREA_MAX = 30000
CIRCLE_AREA_MIN = 10000
CIRCLE_RATE_MAX = 0.01

# 指针轮廓
POINTER_AREA_MAX = 2000
POINTER_AREA_MIN = 200
POINTER_RATE1 = 3
POINTER_RATE2 = 0.3

# hsv阈值
HSV_RED_L1 = np.array([0, 43, 46])
HSV_RED_H1 = np.array([20, 255, 255])
HSV_RED_L2 = np.array([146, 43, 46])
HSV_RED_H2 = np.array([180, 255, 255])
HSV_GREEN_L = np.array([25, 43, 46])
HSV_GREEN_H = np.array([87, 255, 255])
HSV_YELLOW_L = np.array([16, 43, 46])
HSV_YELLOW_H = np.array([44, 255, 255])
PIXEL_MIN = 200

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
    return None

class PlateDetect:
    def __init__(self):
        # 创建表盘和指针的轮廓
        self.circle_contour = self.create_circle_contour()
        self.pointer_contour = self.create_pointer_contour()
        
        # 圆盘和指针匹配结果轮廓
        self.circle_contour_result = None
        self.pointer_contour_result = None

        # 圆盘和指针匹配是否有结果标志
        self.circle_flag = None
        self.pointer_flag = None

        # 圆盘轮廓圆形与半径
        self.circle_radius = None
        self.circle_center = None

        # 指针指向点
        self.point_color = None
        self.point_color_flag = None

        # 最终结果
        self.result = None

    def create_circle_contour(self):
        circle = np.zeros((100, 100), np.uint8)
        cv.circle(circle, (50, 50), 30, 255, -1)
        circle_contour, _ = cv.findContours(circle,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        return circle_contour

    def create_pointer_contour(self):
        pointer = cv.imread(".\\Media\\pointer.png")
        pointer = cv.cvtColor(pointer, cv.COLOR_BGR2GRAY)
        pointer = cv.GaussianBlur(pointer,(5,5),0)
        _, pointer = cv.threshold(pointer, 250, 255, cv.THRESH_BINARY)
        pointer = cv.bitwise_not(pointer)
        pointer_contour, _ = cv.findContours(pointer,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        return pointer_contour
    
    def cv_image_gray(self, image):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_gray = clahe.apply(image_gray)
        image_gray = cv.GaussianBlur(image_gray,(5,5),0)
        return image_gray

    def cv_image_binary(self ,image_gray):
        # _, image_binary = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, image_binary = cv.threshold(image_gray, 127, 255, cv.THRESH_BINARY)
        image_binary = cv.bitwise_not(image_binary)
        kernel = np.ones((5,5),np.uint8)
        image_binary = cv.morphologyEx(image_binary, cv.MORPH_OPEN, kernel)
        return image_binary

    # 计算两点距离
    def calculation_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # 获取多点与一点，其中最远的那点
    def find_farthest_point(self, center, points):
        max_distance = 0
        farthest_point = points[0]
        for point in points:
            dis = self.calculation_distance(center, point)
            if dis > max_distance:
                max_distance = dis
                farthest_point = point
        return farthest_point, max_distance

    # 表盘轮廓匹配
    def find_circle_contour(self, image_gray):
        # 查找全部轮廓
        contours, _ = cv.findContours(image_gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        contours_result = []
        rates_result = []
        # 遍历全部轮廓
        for i, contour in enumerate(contours):
            # 轮廓面积筛选
            area = cv.contourArea(contour)
            if area > CIRCLE_AREA_MIN and area < CIRCLE_AREA_MAX:
                # 匹配圆形轮廓
                rate = cv.matchShapes(contour, self.circle_contour[0], cv.CONTOURS_MATCH_I1, 0)
                # 筛选离谱的结果
                if (rate < CIRCLE_RATE_MAX):
                        contours_result.append(contour)
                        rates_result.append(rate)

        rates_result = np.array(rates_result)
        if len(rates_result) > 0:
            _, _, min_locate, _ = cv.minMaxLoc(rates_result)
            contour_result = contours_result[min_locate[1]]
            return True, contour_result
        else:
            return False, None
        
    # 表盘图像提取
    def get_circle_area(self, image_gray):
        # 轮廓近似为圆
        (cir_x, cir_y), cir_r = cv.minEnclosingCircle(self.circle_contour_result)
        cir_c = np.array([int(cir_x), int(cir_y)])
        self.circle_center = cir_c
        self.circle_radius = int(cir_r)
        cir_r = int(cir_r * 0.8)

        # 创建圆形掩膜，去除背景
        mask = np.zeros_like(image_gray)
        cv.circle(mask, cir_c, cir_r, 255, -1)
        return cv.bitwise_and(image_gray, mask)
        
    # 指针轮廓匹配
    def fing_pointer_contour(self, image_gray):
        # 查找全部轮廓
        contours, _ = cv.findContours(image_gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        contours_result = []
        rates_result = []
        # 遍历全部轮廓
        for i, contour in enumerate(contours):
            # 轮廓面积筛选
            area = cv.contourArea(contour)
            if area > POINTER_AREA_MIN and area < POINTER_AREA_MAX:
                # 长宽比筛选
                rect = cv.minAreaRect(contour)
                w, h = rect[1]
                pointer_rate = w / h
                if pointer_rate > POINTER_RATE1 or pointer_rate < POINTER_RATE2:
                    # 匹配指针轮廓
                    rate = cv.matchShapes(contour, self.pointer_contour[0], cv.CONTOURS_MATCH_I1, 0)
                    contours_result.append(contour)
                    rates_result.append(rate)

        rates_result = np.array(rates_result)
        if len(rates_result) > 0:
            _, _, min_locate, _ = cv.minMaxLoc(rates_result)
            contour_result = contours_result[min_locate[1]]
            return True, contour_result
        else:
            return False, None

    # 指针指向区域提取
    def get_pointer_area(self, image_bgr):
        # 对指针轮廓进行简化
        epsilon = 0.05*cv.arcLength(self.pointer_contour_result,True)
        self.pointer_contour_result = cv.approxPolyDP(self.pointer_contour_result,epsilon,True)        

        # 获取指针轮廓距离圆形最远一点
        point_farthest, distance_max = self.find_farthest_point(self.circle_center, self.pointer_contour_result[0])

        # 获取表盘区域一点
        # 筛选较短距离
        if distance_max > self.circle_radius / 4:
            ratio = (self.circle_radius / 1.3) / distance_max
            point_color = (point_farthest - self.circle_center)* ratio + self.circle_center
            self.point_color = np.array(point_color, np.int32)

            # 以该点为圆形建立掩膜，截取指针指向区域
            mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            cv.circle(mask, self.point_color, int(self.circle_radius/4), (255,255,255), -1, 0)

            return True, cv.bitwise_and(image_bgr, image_bgr, mask = mask)
        else:
            return False, None
 
    # 使用HHSV色彩空间进行阈值分割并计数像素点
    def count_hsv_red(self, image_hsv):
        mask1 = cv.inRange(image_hsv, HSV_RED_L1, HSV_RED_H1)
        mask2 = cv.inRange(image_hsv, HSV_RED_L2, HSV_RED_H2)
        mask = cv.bitwise_or(mask1, mask2)
        nz_count = cv.countNonZero(mask)
        return nz_count
    def count_hsv_green(self, image_hsv):
        mask = cv.inRange(image_hsv, HSV_GREEN_L, HSV_GREEN_H)
        nz_count = cv.countNonZero(mask)
        return nz_count
    def count_hsv_yellow(self, image_hsv):
        mask = cv.inRange(image_hsv, HSV_YELLOW_L, HSV_YELLOW_H)
        nz_count = cv.countNonZero(mask)
        return nz_count

    # 最终判断结果
    def detect_color(self, image_hsv):
        non_zero_r = self.count_hsv_red(image_hsv)
        non_zero_g = self.count_hsv_green(image_hsv)
        non_zero_y = self.count_hsv_yellow(image_hsv)
        nz_rgy = np.array([non_zero_r, non_zero_g, non_zero_y])
        # print(nz_rgy)
        _, _, _, nz_max_loc = cv.minMaxLoc(nz_rgy)
        if nz_rgy[nz_max_loc[1]] > PIXEL_MIN:
            if nz_max_loc[1] == 0:
                return "red"
            elif nz_max_loc[1] == 1:
                return "green"
            elif nz_max_loc[1] == 2:
                return "yellow"
        else:
            return None


    # 主循环中运行
    def run(self, frame):
        # 图像预处理
        frame_g = self.cv_image_gray(frame)
        frame_b = self.cv_image_binary(frame_g)

        self.circle_flag, self.circle_contour_result = self.find_circle_contour(frame_b)
        if self.circle_flag:
            frame_b_cut = self.get_circle_area(frame_b)
            self.pointer_flag, self.pointer_contour_result = self.fing_pointer_contour(frame_b_cut)
            if self.pointer_flag:
                self.point_color_flag, frame_bgr_cut = self.get_pointer_area(frame)
                if self.point_color_flag:
                    frame_hsv_cut = cv.cvtColor(frame_bgr_cut, cv.COLOR_BGR2HSV)
                    self.result = self.detect_color(frame_hsv_cut)
                else:
                    self.result = None
            else:
                self.result = None
        else:
            self.result = None



if __name__ == "__main__":
    # 创建队列缓冲，对结果进行滤波
    result_buffer = queue.Queue(maxsize = BUFFER_LEN)

    plate = PlateDetect()

    # 计算帧率
    frame_count = 0
    fps = 0
    start_time = time.time()

    cap = cv.VideoCapture(".\\Media\\video1.mp4")
    while True:

        ret, frame = cap.read()
        if not ret:
            print(f"Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.resize(frame, (WIDTH, HIGH))
        plate.run(frame)

        # 轮廓绘制
        if plate.circle_flag:
            # 绘制表盘轮廓
            # cv.drawContours(frame, [plate.circle_contour_result], 0, LINE_COLOR, 2)
            cv.circle(frame, plate.circle_center, plate.circle_radius, LINE_COLOR, 2, 0)
            if plate.pointer_flag:
                # 绘制指针轮廓
                cv.drawContours(frame, [plate.pointer_contour_result], 0, LINE_COLOR, 2)
                if plate.pointer_flag:
                    # 绘制颜色判断区轮廓
                    cv.circle(frame, plate.point_color, int(plate.circle_radius/4), LINE_COLOR, 2, 0)
        
        # 结果滤波
        if result_buffer.full():
            result_buffer.get()
        result_buffer.put(plate.result)
        result = check_queue(result_buffer)
        # 打印输出最终结果
        if result != None:
            if result == "green":
                print(f"正常、正常状态")
            elif result == "yellow":
                print(f"偏低、异常状态")
            elif result == "red":
                print(f"偏高、异常状态")

        # 计算帧率
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1.0:
            fps = frame_count
            frame_count = 0
            start_time = current_time
        cv.putText(frame, f"FPS:{fps}", (5, 30), cv.FONT_HERSHEY_SIMPLEX, 1, LINE_COLOR, 2, cv.LINE_AA)

        cv_imshow("frame", frame)

        if cv.waitKey(10) == 27:
            break
    cap.release()
    cv.destroyAllWindows()










