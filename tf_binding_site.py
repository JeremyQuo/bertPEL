import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import numpy as np

#读取表达量数据
count_df = pd.read_csv('data/pa.count')
count_dict = count_df.set_index('gene_name')['expression'].to_dict()


# 读取TF数据
df = pd.read_csv('data/wrapped_data_pspa.csv')  # 替换为你的数据文件名
# 数据预处理
label_encoder_tf = LabelEncoder()
label_encoder_feature = LabelEncoder()
df['tf_encoded'] = label_encoder_tf.fit_transform(df['TFname'])
df['feature_encoded'] = label_encoder_feature.fit_transform(df['bindfeature'])



# 读取mod数据
mod_df = pd.read_csv('data/pa_T1motif.csv')  # 替换为你的数据文件名
mod_df['feature_encoded'] = label_encoder_feature.transform(mod_df['modfeature'])
df_grouped = mod_df.groupby(['modgene', 'feature_encoded']).size().reset_index(name='counts')
df_grouped = df_grouped.groupby('modgene')
mod_dict={}
max_pos=0
for name, group in df_grouped:
    mod_dict[name] = group[['feature_encoded', 'counts']]
    if group.shape[1] > max_pos:
        max_pos = group.shape[1]

df_grouped = df.groupby('bindgene')

data_list = []
max_len=0
for name, group in df_grouped:
    if name not in count_dict:
        continue
    tf_name = torch.tensor(group['tf_encoded'].values, dtype=torch.long)
    bind_feature = torch.tensor(group['feature_encoded'].values, dtype=torch.long)
    combined = torch.stack([tf_name, bind_feature], dim=0)
    # prepare mod feature
    if name not in mod_dict:
        mod_x= torch.zeros(2, max_pos)
    else:
        mod_info=mod_dict[name]
        mod_num = torch.tensor(mod_info['counts'].values, dtype=torch.long)
        bind_feature = torch.tensor(mod_info['feature_encoded'].values, dtype=torch.long)
        mod_combined = torch.stack([mod_num, bind_feature], dim=0)
        mod_x = torch.cat([mod_combined, torch.zeros(2, max_pos - mod_combined.shape[1])], dim=1)
    y = torch.tensor(count_dict[name], dtype=torch.float)
    data = Data(tf_x=combined,mod_x=mod_x, y=y)
    if group['tf_encoded'].shape[0] > max_len:
        max_len = group['tf_encoded'].shape[0]
    data_list.append(data)

def collate_fn(batch):
    global max_len
    sequences,temp, targets = zip(*[(data.tf_x,data.mod_x, data.y) for data in batch])
    padded_sequences = [torch.cat([seq, torch.zeros(2, max_len - seq.shape[1])], dim=1) for seq in sequences]
    targets = torch.stack(targets)
    temp = torch.stack(temp)
    return torch.stack(padded_sequences),temp, targets

train_dataset = DataLoader(data_list, batch_size=1000, shuffle=True, collate_fn=collate_fn)

# 定义DNN模型
class DNNModel(nn.Module):
    global max_len
    def __init__(self):
        super(DNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        self.fc1 = nn.Linear(max_len, 128)  # 输入特征数为2（tf_name, bind_feature）* max_len
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 4)
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        self.fc3 = nn.Linear(max_pos, 16)  # 输入特征数为2（tf_name, bind_feature）* max_len
        self.dropout = nn.Dropout(0.1)
        self.fc4 = nn.Linear(16, 4)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, inputs,inputs_mod):
        x = self.conv1(inputs)
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        x_mod = self.conv2(inputs_mod)
        x_mod = x_mod.squeeze(1)
        x_mod = F.relu(self.fc3(x_mod))
        x_mod = self.dropout(x_mod)
        x_mod = self.fc4(x_mod)

        x = torch.cat((x, x_mod), dim=1)
        return F.relu(self.fc5(x))

# 创建模型、优化器和损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DNNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss().to(device)

loss_list=[]
# 训练模型
for epoch in range(2000):  # Number of epochs
    model.train()
    for batch in train_dataset:
        inputs_tf,input_mod, targets = batch
        inputs_tf,input_mod, targets = inputs_tf.to(device), input_mod.to(device),targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs_tf.float(),input_mod)
        loss = criterion(predictions.squeeze(), targets)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {np.mean(loss_list)}")
print(1)