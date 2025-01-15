import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from collections import OrderedDict

device = torch.device('cuda:0')

def add_spaces(input_string):
    # 使用列表推导式将字符串每三个字符分组
    spaced_string = ' '.join(input_string[i:i+3] for i in range(0, len(input_string), 3))
    return spaced_string

def read_fasta_to_dic(filename):
    """
    function used to parser small fasta
    still effective for genome level file
    """
    fa_dic = OrderedDict()

    with open(filename, "r") as f:
        for n, line in enumerate(f.readlines()):
            if line.startswith(">"):
                if n > 0:
                    fa_dic[short_name] = "".join(seq_l)  # store previous one

                full_name = line.strip().replace(">", "")
                short_name_list = full_name.split(";")
                status = False
                for item in short_name_list:
                    if 'locus' in item:
                        short_name=item.split("=")[1]
                        status = True
                if not status:
                    print(1)
                seq_l = []
            else:  # collect the seq lines
                if len(line) > 8:  # min for fasta file is usually larger than 8
                    seq_line1 = line.strip()
                    seq_l.append(seq_line1)

        fa_dic[short_name] = "".join(seq_l)  # store the last one
    return fa_dic

def save_fasta_dict(fasta_dict,path):
    f=open(path,'w+')
    for key,value in fasta_dict.items():
        f.write('>'+key+'\n')
        line = len(value) // 80 + 1
        for i in range(0, line):
            f.write(value[i*80:(i+1)*80]+'\n')
    f.close()



class MyDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        encoding = self.tokenizer(sequence, padding='max_length', truncation=True, return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), torch.tensor(target,
                                                                                                   dtype=torch.float)


class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.to(device)
        self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token's output
        prediction = self.regressor(cls_output)
        return prediction

# Check if CUDA is available and set device

count_df = pd.read_csv('data/pa.count')
count_dict = count_df.set_index('gene_name')['expression'].to_dict()
# Example sequences and targets
sequence_dic = read_fasta_to_dic("data/pa.ffn")

sequences=[]
targets=[]
for key, value in count_dict.items():
    sequences.append(add_spaces(sequence_dic[key]))
    targets.append(count_dict[key])


dataset = MyDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=25, shuffle=True)

model = RegressionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.L1Loss().to(device)

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    for batch in dataloader:
        input_ids, attention_mask, targets = batch
        input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions.squeeze(), targets)
        print(loss)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Example prediction

with torch.no_grad():
    model.eval()
    sample_sequence = "New sequence to predict"
    encoding = dataset.tokenizer(sample_sequence, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    prediction = model(input_ids, attention_mask)
    print(f"Prediction: {prediction.item()}")
