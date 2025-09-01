import torch
import torch.nn as nn
import transformers
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import get_polynomial_decay_schedule_with_warmup
import os
import ujson
import pdb
import datetime
import logging

num_epochs = 3
batch_size = 48
lr = 1e-5
train_eval_split = 0.9
num_warmup_steps = 2000
num_training_steps = 30000
log_step = 50
eval_step = 2000
hidden_size = 512

logging.basicConfig(filename=f'final_base_{hidden_size}_256_{lr}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = "/mnt/models/zks/bge-base-en-v1.5"
device = 'cuda:3'

class BertForRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)
        self.regression = torch.nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            #nn.Linear(hidden_size, 1)
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )


    def forward(self, inputs):
        encoded = self.bert(**inputs)
        score = self.regression(encoded['pooler_output'])
        return encoded, score




def calculate_accuracy(outputs, labels):
    preds = (outputs >= 0).float()
    accuracy = (preds == labels).float().mean().item()
    return accuracy

def configure_optimizers(model):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW([p for n, p in model.named_parameters() if p.requires_grad],
                        lr=lr,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=1e-2)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=0,
        power=1,
    )
    return loss_fn, optimizer, lr_scheduler

def training_step(model, tokenizer, batch, loss_fn):
    text1, text2, labels = batch
    tokenized_text1 = tokenizer(text1, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    tokenized_text2 = tokenizer(text2, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    labels = labels.to(device)

    encoded1, score1 = model(tokenized_text1)
    encoded2, score2 = model(tokenized_text2)

    output = torch.sub(score2, score1).squeeze()
    loss = loss_fn(output, labels)
    acc = calculate_accuracy(output, labels)
    return loss, acc

def evaluation_step(model, tokenizer, batch, loss_fn):
    text1, text2, labels = batch
    tokenized_text1 = tokenizer(text1, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    tokenized_text2 = tokenizer(text2, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    labels = labels.to(device)
    
    encoded1, score1 = model(tokenized_text1)
    encoded2, score2 = model(tokenized_text2)

    output = torch.sub(score2, score1).squeeze()
    loss = loss_fn(output, labels)
    acc = calculate_accuracy(output, labels)
    return loss, acc


def train_model(model, tokenizer, train_loader, eval_loader, loss_fn, optimizer, lr_scheduler):
    tot_step = 0
    min_eval_loss = 1e5

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_step = 0
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            loss, acc = training_step(model, tokenizer, batch, loss_fn)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_step += 1
            tot_step += 1
            epoch_loss += loss.item()
            epoch_accuracy += acc


            if epoch_step % log_step == 1:   
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")}--Epoch {epoch+1} - Step {epoch_step} - Training loss: {loss:.4f}, Training accuracy: {acc:.4f}')
                logging.info(f'{datetime.datetime.now().strftime("%H:%M:%S")}--Epoch {epoch+1} - Step {epoch_step} - Training loss: {loss:.4f}, Training accuracy: {acc:.4f}')

            if tot_step % eval_step == 0:
                model.eval()
                total_eval_loss = 0
                total_eval_accuracy = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        eval_loss, eval_acc = evaluation_step(model, tokenizer, eval_batch, loss_fn)
                        total_eval_loss += eval_loss.item()
                        total_eval_accuracy += eval_acc

                avg_eval_loss = total_eval_loss / len(eval_loader)
                avg_eval_acc = total_eval_accuracy / len(eval_loader)

                print(f'{datetime.datetime.now().strftime("%H:%M:%S")}----------Total step {tot_step} - Evaluation loss: {avg_eval_loss:.4f}, Evaluation accuracy: {avg_eval_acc:.4f}')
                logging.info(f'{datetime.datetime.now().strftime("%H:%M:%S")}----------Total step {tot_step} - Evaluation loss: {avg_eval_loss:.4f}, Evaluation accuracy: {avg_eval_acc:.4f}')


                if avg_eval_loss < min_eval_loss:  # save model with the best evaluation loss
                    min_eval_loss = avg_eval_loss
                    if avg_eval_acc > 0.8:
                        print(f'{datetime.datetime.now().strftime("%H:%M:%S")}----------Saving model with loss {avg_eval_loss:.4f}----------')
                        logging.info(f'\n{datetime.datetime.now().strftime("%H:%M:%S")}----------Saving model with loss {avg_eval_loss:.4f}----------\n')
                        torch.save(model.state_dict(), f'/mnt/models/zks/bge-base-en-v1.5/pairwise_params/{hidden_size}_256_step{tot_step}-loss{avg_eval_loss:.4f}-acc{avg_eval_acc:.4f}.pt')


        avg_train_loss = epoch_loss / epoch_step
        avg_train_acc = epoch_accuracy / epoch_step

        print(f'{datetime.datetime.now().strftime("%H:%M:%S")}*********Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}, Average training accuracy: {avg_train_acc:.4f}')
        logging.info(f'{datetime.datetime.now().strftime("%H:%M:%S")}*********Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}, Average training accuracy: {avg_train_acc:.4f}') 



def prepare_data():
    data_dir = '/mnt/data/zks/train_pairwise/en/annotation/'
    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
    data = []

    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                item = ujson.loads(line)
                if item['result'] != 'Text0' and item['result'] != 'Text1':
                    raise ValueError(f"Invalid label: {item['label']}")
                data.append((item['text1'], item['text2'], 1.0 if item['result'] == 'Text1' else 0.0))

    print(f'Dataset size: {len(data)}')
    return list(zip(*data))

class PairDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx], self.data[2][idx]

def build_dataloader(data, batch_size, train_eval_split):
    dataset = PairDataset(data)
    train_size = int(train_eval_split * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader

def main():
    data = prepare_data()
    train_loader, eval_loader = build_dataloader(data, batch_size, train_eval_split)
    print('Data prepared. Start training...')

    model = BertForRegression(model_name)
    tokenizer = transformers.BertTokenizerFast.from_pretrained(model_name)

    model = model.to(device)

    loss_fn, optimizer, lr_scheduler = configure_optimizers(model)

    train_model(model, tokenizer, train_loader, eval_loader, loss_fn, optimizer, lr_scheduler)

if __name__ == "__main__":
    main()