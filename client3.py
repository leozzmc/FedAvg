from ultralytics import YOLO
from Fedclient_2 import FedClient
import os
import torch
import matplotlib.pyplot as plt

iterations = 5  # 要進行的聯邦學習次數
modelcount = 2  # 模型的數量 (假設有2個模型)
clientId = 1  # 客戶端ID
epochs_client = 100  # 本地訓練的迭代次數
imgsz = 640  # 圖像大小
batch_size = 16  # 批量大小
datasets = [
    "/mnt/c/Users/Kevin/FedAvg/clients/client2/horizon/data.yaml",
    "/mnt/c/Users/Kevin/FedAvg/clients/client2/top/data.yaml",
]

global_weights_file = 'downloaded_global_weights.pth'
accuracy_trend = []  # 存儲每次迭代的準確度

if __name__ == "__main__":
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
    fedclient = FedClient()

    for iteration in range(1, iterations + 1):
        iteration_accuracies = []
        for i in range(modelcount):
            # 在本地訓練模型並將權重保存到文件中
            weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, clientId)
            
            # 上傳本地權重文件到服務器
            upload_response = fedclient.upload_weights(clientId, weights_file)
            
            if upload_response.get('status') == 'success':
                # 從服務器下載全局權重
                if fedclient.download_global_weights():
                    # 將全局權重加載到模型中
                    global_weights = torch.load(global_weights_file)
                    models[i].load_state_dict(global_weights)

                    # 評估更新後的模型
                    local_eval = fedclient.evaluate_model(models[i], datasets[i], iteration, clientId)
                    if 'accuracy' in local_eval:
                        iteration_accuracies.append(local_eval['accuracy'])
                    print(f"Client {i} 本地評估結果: {local_eval}")

        # 確保不會因為迭代準確度為空而導致除以零的錯誤
        if iteration_accuracies:
            accuracy_trend.append(sum(iteration_accuracies) / len(iteration_accuracies))

    # 繪製迭代後的準確度趨勢圖
    if accuracy_trend:
        plt.plot(range(1, len(accuracy_trend) + 1), accuracy_trend, marker='o')
        plt.title(f"Client {clientId} 準確度趨勢 (共 {iterations} 次迭代)")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.savefig(f'client_{clientId}_accuracy_trend.png')
        plt.show()
    else:
        print("沒有可用的準確度數據進行繪圖。")
