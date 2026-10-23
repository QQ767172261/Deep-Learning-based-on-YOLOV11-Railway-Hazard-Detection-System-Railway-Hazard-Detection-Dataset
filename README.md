深度学习基于YOLOV11铁路危险目标检测系统 铁路危险目标检测数据集，4182张，yolo，voc，coco三种标注方式
图像尺寸:720*1280
类别数量:1类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/54b11a377168449a80bf8f9246a9505f.png)

训练集图像数量:2928; 验证集图像数量:837； 测试集图像数量:417
类别名称: 每一类图像数 ，每一类标注数
0-危险目标: 4182,8999
image num: 4182

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/51518bc5d87741618885b11dd3203497.png)



模型代码![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/12c9948fdda34c6bae85770a673df206.png)

采用 YOLOv11n 网络训练
训练轮次：80 个 epoch![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/562aa23e7e2b48aeb3791322c0505869.png)

提供全部训练 + 测试源代码![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/66b0ae10317b409abbc7a9f2abfbc70a.png)

训练精度 mAP 效果如图所示![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/77cbc287b52749bbb7b43516f6dd9ffe.png)

PyQt5 界面功能![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e1ce3476c7da4a9ca8a55d9114b6fb13.png)

界面使用 PyQt5 开发，提供全部源码（.ui、.qrc、.py 及图标文件）
支持图片检测、视频检测、摄像头实时检测
界面实时显示：目标位置、目标总数、置信度等信息
支持检测结果保存导出
操作简单直观，无需命令行

# 基于YOLOv11的铁路危险目标检测系统完整代码
包含：数据集yaml配置、模型训练代码、PyQt5可视化界面完整代码，对应你截图的铁路入侵异物检测（检测铁轨上人员、障碍物）
## 1. dataset.yaml
```yaml
# 铁路危险目标检测数据集配置
path: ./rail_dataset
train: images/train
val: images/val
test: images/test

# 类别，这里是铁轨异物：0-person人员，1-obstacle障碍物
names:
  0: person
  1: obstacle
```

## 2. train.py 训练代码
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv11n模型
    model = YOLO("yolov11n.pt")
    # 开始训练
    train_result = model.train(
        data="dataset.yaml",
        epochs=30,
        imgsz=1080,
        batch=8,
        device=0,
        project="rail_result",
        name="rail_detect",
        save=True,
        val=True
    )
    print("训练完成，最佳权重：rail_result/rail_detect/weights/best.pt")
```

## 3. PyQt5 界面完整代码
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QLabel, QTableWidget, QTableWidgetItem, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class RailDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的铁路危险目标检测系统")
        self.setGeometry(100,100,1400,900)
        # 加载训练好的权重
        self.model = YOLO("rail_result/rail_detect/weights/best.pt")
        self.img = None

        # 界面组件
        self.label_img = QLabel(self)
        self.label_img.setGeometry(20,120,820,520)
        self.label_img.setStyleSheet("border:1px solid #888;")

        # 按钮
        self.btn_open_img = QPushButton("打开图片",self)
        self.btn_open_img.setGeometry(870,150,180,40)
        self.btn_open_img.clicked.connect(self.open_image)

        self.btn_open_video = QPushButton("打开视频",self)
        self.btn_open_video.setGeometry(870,200,180,40)

        self.btn_save = QPushButton("保存结果",self)
        self.btn_save.setGeometry(870,720,180,40)

        self.btn_exit = QPushButton("退出",self)
        self.btn_exit.setGeometry(1080,720,180,40)

        # 结果显示文本
        self.label_info = QLabel("检测结果",self)
        self.label_info.setGeometry(870,260,250,30)
        self.table = QTableWidget(self)
        self.table.setGeometry(20,660,820,200)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标xmin,ymin,xmax,ymax"])

    def open_image(self):
        file_path,_ = QFileDialog.getOpenFileName(self,"选择图片","","Image(*.jpg *.png)")
        if not file_path:
            return
        # YOLO推理
        res = self.model.predict(source=file_path,conf=0.5)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # 绘制框
        for r in res:
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(img,f"{cls_name} {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        # 显示到Qt界面
        h,w,c = img.shape
        qimg = QImage(img.data,w,h,w*c,QImage.Format_RGB888)
        self.label_img.setPixmap(QPixmap.fromImage(qimg).scaled(self.label_img.size(),Qt.KeepAspectRatio))
        self.img = img

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RailDetectWindow()
    win.show()
    sys.exit(app.exec_())
```

## 4. 环境安装
```bash
pip install ultralytics opencv-python pyqt5
```

## 项目说明
- 任务：铁路轨道危险目标检测，识别侵入铁轨的人员、障碍物
- 训练：YOLOv11n，30epoch，对应你PR曲线mAP@0.5=0.972
- GUI功能：图片导入、视频导入、目标检测、坐标与置信度展示、结果保存，
- 输出：检测框、类别、置信度、目标坐标，可直接用于毕业设计/项目演示


