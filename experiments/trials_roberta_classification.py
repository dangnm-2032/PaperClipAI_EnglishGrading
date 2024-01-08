from typing import Any, Dict, List, Optional, Tuple, Union
from torch._tensor import Tensor
from torch.nn.modules import Module
from transformers import RobertaModel, PreTrainedModel
from torch import nn
from transformers import Trainer, AutoTokenizer, AutoConfig, TrainingArguments, DataCollatorWithPadding
import wandb
import torch
from unstructured.cleaners.core import clean_extra_whitespace
from datasets import load_dataset, Dataset

class CustomROBERTAModel(PreTrainedModel):
    def __init__(self, config, base_model, num_feats):
        super(CustomROBERTAModel, self).__init__(config)
        self.roberta = RobertaModel.from_pretrained(
                base_model,
                config=config
        )
        ### New layers:
        self.classification_layer = nn.Sequential(
                nn.Linear(config.hidden_size, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 54),
                nn.GELU(),
                nn.Unflatten(1, (6, 9)),
                nn.Softmax(dim=2),
        )

    def forward(self, **inputs):
        roberta_outputs = self.roberta(**inputs)
        logits = self.classification_layer(roberta_outputs.pooler_output)
        return logits

    def _init_weights(self, module):
        self.roberta._init_weights(module)

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
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
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

def get_ohe_transform():
    ohe_transform = {}
    for i in range(2, 11):
        ohe = [0] * 9
        ohe[i-2] = 1
        ohe_transform[i/2] = ohe
    return ohe_transform

def transform(batch):
    tokenized_input = tokenizer(batch['full_text'], return_tensors='pt', truncation=True)
    input_ids = tokenized_input['input_ids'][0]
    attention_mask = tokenized_input['attention_mask'][0]
    targets_feat = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    targets = []
    ohe_transform = get_ohe_transform()
    for feat in targets_feat:
        targets.append(
            torch.tensor(
                ohe_transform[
                    batch[feat]
                ],
                dtype=torch.float32
            )
        )
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'target': torch.stack(targets)
    }

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
base_model = "roberta-large"
config = AutoConfig.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = CustomROBERTAModel(
    config=config, 
    base_model=base_model,
    num_feats=6
) # You can pass the parameters if required to have more flexible model
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
    "num_train_epochs": 50,
    "log_level": "error",
    "logging_steps": 1,
    "report_to": "wandb",
    "full_determinism": False,
    'save_strategy': 'no',

}
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    remove_unused_columns=False,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    warmup_ratio=0.01,
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
save_name = "EnglishGradingModel"
trainer.save_model(save_name)
config.save_pretrained(save_name)
tokenizer.save_pretrained(save_name)
