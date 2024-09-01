import os
import matplotlib.pyplot as plt
import torch
import requests

class FedClient:
    
    def __init__(self) -> None:
        self.weights_file_template = 'client_{client_id}_weights.pth'
        self.global_weights_file = 'downloaded_global_weights.pth'
        self.accuracy_history = {}

    def plot_accuracy_trend(self, client_id, iteration, accuracy, save_dir='accuracy_trends'):
        # 初始化客戶端的準確度歷史記錄（如尚未存在）
        if client_id not in self.accuracy_history:
            self.accuracy_history[client_id] = []

        # 更新客戶端的準確度歷史記錄
        self.accuracy_history[client_id].append((int(iteration), accuracy))

        # 繪製折線圖
        plt.figure()
        iterations, accuracies = zip(*self.accuracy_history[client_id])
        plt.plot(iterations, accuracies, marker='o', linestyle='-', color='b')
        plt.title(f'聯邦學習的準確度趨勢 (Client {client_id})')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)

        # 如不存在則創建目錄
        os.makedirs(save_dir, exist_ok=True)

        # 保存繪圖
        save_path = os.path.join(save_dir, f'accuracy_trend_client_{client_id}.png')
        plt.savefig(save_path)
        plt.close()

    def evaluate_model_logic(self, model, dataset_path):
        # 加載數據集並進行評估
        results = model.val(data=dataset_path)  # `val` 函數通常用於評估
        
        # 提取準確度或相關指標
        accuracy = results.box.map  # 平均精度（mAP）
        
        return {"accuracy": accuracy}

    def evaluate_model(self, model, dataset_path, iteration, client_id):
        accuracy_result = self.evaluate_model_logic(model, dataset_path)  # 直接傳遞文件路徑
        accuracy = accuracy_result['accuracy']
        
        # 更新並保存準確度趨勢
        self.plot_accuracy_trend(client_id=client_id, iteration=iteration, accuracy=accuracy)

        return accuracy_result

    def train_on_client(self, model, dataset_path, epochs, batch_size, client_id):
        model.train(data=dataset_path, epochs=epochs, batch=batch_size, save=False)
        local_weights = model.state_dict()  # 獲取訓練後的模型權重
        # 將權重保存到文件中
        weights_file = self.weights_file_template.format(client_id=client_id)
        torch.save(local_weights, weights_file)
        return weights_file

    def upload_weights(self, client_id, weights_file):
        url = f"http://localhost:5000/api/upload_weights/{client_id}"
        with open(weights_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)

        if response.status_code != 200:
            print(f"無法上傳客戶端 {client_id} 的權重: {response.status_code}")
            print(response.text)
            return {"status": "error"}

        return response.json()

    def download_global_weights(self):
        url = "http://localhost:5000/api/download_global_weights"
        response = requests.get(url)

        if response.status_code == 200:
            with open(self.global_weights_file, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"無法下載全局權重: {response.status_code}")
            print(response.text)
            return False
