# 蘭花照片處理說明


## YOLOv8 模型訓練

 - `Orchid_label.v2i.yolov8/`: 只為蘭花俯拍照片 （`top` folder on the FTP server）而train 的模型。
 - `Orchid_label.v3i.yolov8/`: 只為蘭花平拍照片 （`horizon` folder on the FTP server）而train 的模型。
 - `Orchid_label.v4i.yolov8/`: 只為蘭花平拍照片 （`horizon` & `top`, and resize the image to 64x64）而train 的模型。

## 步驟

- 先在 Roboflow 上對蘭花照片進行 labeling ，俯拍照片使用方框框選葉片，平拍照片根據葉片形狀而個別匡選
- 匯出 YOLOv8 資料集
- 解壓縮至專案中
- 變更 `data.yaml` 中的 `train`, `test`,`val` 各別路徑
- 執行 yolo 指令

```
yolo train data=data.yaml model=yolov8n.pt epochs=50 batch=16 
```
- 完成後可以執行預測

```
yolo detect predict model=runs/detect/train2/weights/best.pt source=valid/images save=true device=cpu
```
預測結果會在 `runs/predict` 目錄底下


## 目前測試結果

- `Orchid_label.v2i.yolov8/` Prediction:
  - ![79_JPG.rf.7d41e42acd808ca6d75395fb98686f8b](https://hackmd.io/_uploads/rk3fyMEj0.jpg)
- `Orchid_label.v3i.yolov8/` Prediction:
  - ![24_1_JPG.rf.7d1b6cfe48c914774708897520111b47](https://hackmd.io/_uploads/rkZRrMEiC.jpg)
- `Orchid_label.v4i.yolov8/` Prediction:
  - ![62_JPG.rf.df695a54da88a2a353dc396b787af4a3](https://hackmd.io/_uploads/BySywzNoC.jpg)
  - ![95_1_JPG.rf.4ccd29b0461965bcb9bb93f9d4ecad11](https://hackmd.io/_uploads/rJqxDzNsR.jpg)

## 下一步

- `client1/`: 模擬大公司，佔有 70% 的原始 datasets
  - 已經建立好模型跟進行預測
  - 有寫一個  `count_number.py` 可以去load train 好的模型然後去計算葉子數量，其實就是去看有幾個label為 `leaf` bounding box 就算是這個模型辨識出了幾個葉子
- `client2/`: 模擬小公司，佔有 30% 的原始 datasets
  - 已經建立好模型跟預測