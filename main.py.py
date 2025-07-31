import nltk
import os
import shutil
import torch 
import torch as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer,BertForSequenceClassification
import json
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset


#1.json資料前處理
file="C:/Users/user/Desktop/CM-1.json"

with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)


with open('output.jsonl', 'w',encoding='utf-8') as outfile:
    for entry in data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')
        

classification_data = []
for item in data:
    entry = {
        "text": item["question"],
        "label": item["category"]
    }
    classification_data.append(entry)

print(entry)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
label_map = {'中藥學':0,
          '方劑學':1
           }





#2.資料切分

random.seed(42)

# 按比例分割
train_files, temp_files = train_test_split(classification_data, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)


print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
class CMDataset(Dataset) :
      def __init__(self, data, label_map,tokenizer):
        self.data =data
        self.len = len(self.data)
        self.label_map = label_map
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
      def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label =self.label_map[item["label"]]
        encoding = self.tokenizer(
        text,
        padding='max_length',      # 補到固定長度
        truncation=True,           # 太長就截斷
        max_length=128,            # 你設定的最大長度
        return_tensors='pt')       # 直接回傳 tensor 格式
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)
        label_tensor = torch.tensor(label)

        return {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "token_type_ids": token_type_ids,
    "label": label_tensor
}
    
      def __len__(self):
        return self.len
            
train_dataset =CMDataset(train_files, label_map, tokenizer)
val_dataset = CMDataset(val_files, label_map, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2,shuffle=True)



model = BertForSequenceClassification.from_pretrained("baseten/BertForSequenceClassificationTesting")
epochs = 5


  # 判斷是否使用GPU
    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
    # 定義損失函數和優化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-5)
if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # 開始進入訓練循環
for epoch_num in range(epochs):
            total_acc_train = 0
            total_count_train = 0
            total_loss_train = 0
                
      # 進度條函數tqdm
            
            for batch in tqdm(train_dataloader):
                 mask = batch['attention_mask'].to(device)
                 input_ids = batch["input_ids"].to(device)
                 labels = batch["label"].to(device) 
        # 通過模型得到輸出
                 output = model(input_ids=input_ids, attention_mask=mask)
                # 計算損失
                 batch_loss = criterion(output.logits, labels)
                 total_loss_train += batch_loss.item()
                # 計算精度
                 total_acc_train += (output["logits"].argmax(dim=1) == labels).sum().item()   #計算預測隊的樣本數
                 total_count_train += labels.size(0)    #紀錄總共樣本數
            
        # 模型更新
                 model.zero_grad()
                 batch_loss.backward()
                 optimizer.step()
            # ------ 驗證模型 -----------
            # 定義兩個變量，用於存儲驗證集的準確率和損失
                 total_acc_val = 0
                 total_loss_val = 0
      # 不需要計算梯度
            with torch.no_grad():
                total_count_val = 0
                # 循環獲取數據集，並用訓練好的模型進行驗證
                for batch in val_dataloader:
          # 如果有GPU，則使用GPU，接下來的操作同訓練
                    val_label = batch["label"].to(device)
                    mask = batch['attention_mask'].to(device)
                    input_id = batch['input_ids'].to(device)
  
                    output = model(input_ids=input_id, attention_mask=mask)

                    batch_loss = criterion(output.logits, val_label)
                    total_loss_val += batch_loss.item()
                    
                    
                    total_acc_val += (output.logits.argmax(dim=1) == val_label).sum().item()
                    total_count_val += val_label.size(0)
            
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_dataloader): .3f} 
              | Train Accuracy: {total_acc_train/ total_count_train : .3f} 
              | Val Loss: {total_loss_val / len(val_dataloader): .3f} 
              | Val Accuracy: {total_acc_val / total_count_val: .3f}''')



