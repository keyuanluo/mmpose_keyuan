import cv2
import numpy as np

#---------------------------
# 1. 读取左图（原图）和右图（被裁剪的图）
#---------------------------
left_img_path = "/home/robert/mmpose/demo/masterarbeit_1.png"   # 左图路径
right_img_path = "/home/robert/mmpose/demo/masterarbeit_01_pose.jpg" # 右图路径

left_img = cv2.imread(left_img_path)
right_img = cv2.imread(right_img_path)

#---------------------------
# 2. 目标矩形区域的坐标（浮点数取整）
#   x1,y1 为左上角，x2,y2 为右下角
#---------------------------
x1 = 855.66732732833
y1 = 526.460106999459
x2 = 883.959613874342
y2 = 638.118628181275

# 将坐标取整(可根据需求选择四舍五入或向下/向上取整)
x1_int = int(round(x1))
y1_int = int(round(y1))
x2_int = int(round(x2))
y2_int = int(round(y2))

#---------------------------
# 3. 调整小图尺寸（若被裁剪的图与目标区域大小不一致）
#---------------------------
target_width = x2_int - x1_int
target_height = y2_int - y1_int

# 获取right_img的原始大小
(h, w) = right_img.shape[:2]

# 如果不相等，则进行resize
# （如果 right_img 本来就是从 (x1, y1, x2, y2) 裁下来的，
#   理论上尺寸应与 (target_width, target_height) 相同，
#   这里加上 resize 逻辑只是以防万一）
if (w != target_width) or (h != target_height):
    right_img = cv2.resize(right_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

#---------------------------
# 4. 在left_img上贴回right_img
#---------------------------
# 注意这里使用的是覆盖操作：直接用right_img替换对应区域像素
left_img[y1_int:y1_int+target_height, x1_int:x1_int+target_width] = right_img

#---------------------------
# 5. 保存或显示最终结果
#---------------------------
output_path = "output.jpg"
cv2.imwrite(output_path, left_img)

# 如果要在窗口中显示，可以使用：
# cv2.imshow("result", left_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
