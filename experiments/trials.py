from typing import Any, Dict, List, Optional, Tuple, Union
from torch._tensor import Tensor
from torch.nn.modules import Module
from transformers import BertModel
from torch import nn
from transformers import Trainer, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
import wandb
import torch
from unstructured.cleaners.core import clean_extra_whitespace
from datasets import load_dataset, Dataset

class CustomBERTModel(nn.Module):
      def __init__(self):
            super(CustomBERTModel, self).__init__()
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            ### New layers:
            self.linear1 = nn.Linear(768, 384)
            self.acti1 = nn.LeakyReLU()
            self.linear2 = nn.Linear(384, 192)
            self.acti2 = nn.LeakyReLU()
            self.linear3 = nn.Linear(192, 96)
            self.acti3 = nn.LeakyReLU()
            self.linear4 = nn.Linear(96, 48)
            self.acti4 = nn.LeakyReLU()
            self.linear5 = nn.Linear(48, 24)
            self.acti5 = nn.LeakyReLU()
            self.linear6 = nn.Linear(24, 12)
            self.acti6 = nn.LeakyReLU()
            self.linear7 = nn.Linear(12, 6)
            self.acti7 = nn.LeakyReLU()

      def forward(self, **inputs):
            bert_outputs = self.bert(**inputs)

            # sequence_output has the following shape: (batch_size, sequence_length, 768)
            linear1 = self.linear1(bert_outputs.pooler_output) ## extract the 1st token's embeddings
            acti1 = self.acti1(linear1)
            linear2 = self.linear2(acti1) 
            acti2 = self.acti2(linear2)
            linear3 = self.linear3(acti2) 
            acti3 = self.acti3(linear3)
            linear4 = self.linear4(acti3) 
            acti4 = self.acti4(linear4)
            linear5 = self.linear5(acti4) 
            acti5 = self.acti5(linear5)
            linear6 = self.linear6(acti5) 
            acti6 = self.acti6(linear6)
            linear7 = self.linear7(acti6) 
            acti7 = self.acti7(linear7)
            return acti7

class CustomTrainer(Trainer):
    def __int__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['target']
        # forward pass
        logits = model(
            input_ids=inputs['input_ids'].to('cuda'),
            attention_mask=inputs['attention_mask'].to('cuda'),
        )
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits[-1], labels)
        return (loss, logits) if return_outputs else loss
    def prediction_step(self, model: Module, inputs: Dict[str, Tensor | Any], prediction_loss_only: bool, ignore_keys: List[str] | None = None) -> Tuple[Tensor | None, Tensor | None, Tensor | None]:
         return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
def clean_text(batch):
    text = batch['full_text']
    text = text.replace("\n", ' ')
    text = text.replace("\t", ' ')
    text = text.replace("\r", ' ')
    
    text = clean_extra_whitespace(text)
    batch['full_text'] = text
    return batch

def transform(batch):
    tokenized_input = tokenizer(batch['full_text'], return_tensors='pt', truncation=True)
    input_ids = tokenized_input['input_ids'][0]
    attention_mask = tokenized_input['attention_mask'][0]
    targets_feat = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    targets = []
    for feat in targets_feat:
        targets.append(torch.tensor(batch[feat])/5.0)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'target': targets
    }

def compute_metrics(pred):
    print(pred)
    return None

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="PaperClipAI_EnglishGrading",
    # Track hyperparameters and run metadata
    # config={
    #     "learning_rate": lr,
    #     "epochs": epochs,
    # },
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = CustomBERTModel() # You can pass the parameters if required to have more flexible model
model.to(torch.device("cuda")) ## can be gpu

print("Load DATASET")
dataset = load_dataset('tasksource/english-grading', split='train')
print(dataset)
print("CLEAN")
clean_ds = dataset.map(clean_text)
print(clean_ds)
print("TRANSFORM")
transform_ds = clean_ds.map(transform)
print(transform_ds)
ds = Dataset.from_dict({
    'input_ids': transform_ds['input_ids'],
    'target': transform_ds['target']
}).train_test_split(test_size=0.2)
print(ds)
train_ds = ds['train']
val_ds = ds['test']
print("INIT TRAINING ARGS")
default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "no",
    "num_train_epochs": 10,
    "log_level": "error",
    "logging_steps": 1,
    "report_to": "wandb",
    "full_determinism": False
}
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=4,
    remove_unused_columns=False,
    **default_args)
print("INIT DATA COLLATOR")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("INIT TRAINER")
trainer = CustomTrainer(model=model, 
                        args=training_args,
                        train_dataset=train_ds,
                        # eval_dataset=val_ds,
                        data_collator=data_collator)
trainer.train()