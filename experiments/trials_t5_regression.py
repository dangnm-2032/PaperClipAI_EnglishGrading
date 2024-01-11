from typing import Any, Dict, List, Optional, Tuple, Union
from torch._tensor import Tensor
from torch.nn.modules import Module
from transformers import T5EncoderModel, PreTrainedModel
from torch import nn
from transformers import Trainer, AutoTokenizer, AutoConfig, TrainingArguments, DataCollatorWithPadding
import wandb
import torch
from unstructured.cleaners.core import clean_extra_whitespace
from datasets import load_dataset, Dataset

class CustomT5Model(PreTrainedModel):
    def __init__(self, config, base_model):
        super(CustomT5Model, self).__init__(config)
        self.t5 = T5EncoderModel.from_pretrained(
            base_model,
            config=config
        )
        ### New layers:
        self.regression_layer = nn.Sequential(
            nn.AvgPool2d((1548, 1)),
            nn.Flatten(),
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 6),
            # nn.Sigmoid(),
        )
    def forward(self, **inputs):
        t5_outputs = self.t5(**inputs)
        logits = self.regression_layer(t5_outputs.last_hidden_state)
        return logits

    def _init_weights(self, module):
        self.t5._init_weights(module)

class CustomTrainer(Trainer):
    def __int__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['labels']
        # forward pass
        logits = model(
            input_ids=inputs['input_ids'].to(model.device),
            attention_mask=inputs['attention_mask'].to(model.device),
        )
        loss_fct = nn.MSELoss()
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

def transform(batch):
    tokenized_input = tokenizer(
        batch['full_text'], 
        return_tensors='pt', 
        padding='max_length',
        max_length=1548)
    input_ids = tokenized_input['input_ids'][0]
    attention_mask = tokenized_input['attention_mask'][0]
    targets_feat = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    targets = []
    for feat in targets_feat:
        targets.append(batch[feat]/5.0)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': torch.Tensor(targets)
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
base_model = "google/flan-t5-large"
config = AutoConfig.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = CustomT5Model(
    config=config, 
    base_model=base_model,
) # You can pass the parameters if required to have more flexible model
model.to("cuda") ## can be gpu

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
    'attention_mask': transform_ds['attention_mask'],
    'labels': transform_ds['labels']
}).train_test_split(test_size=0.1).with_format('torch')
print(ds)
train_ds = ds['train']
val_ds = ds['test']
print("INIT TRAINING ARGS")
default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "no",
    "num_train_epochs": 5,
    "log_level": "error",
    "logging_steps": 1,
    "report_to": "wandb",
    "full_determinism": False,
    'save_strategy': 'epoch',

}
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    remove_unused_columns=False,
    gradient_accumulation_steps=32,
    learning_rate=1e-4,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
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
save_name = "artifacts/trained_model/EnglishGrading_t5_regression_5e"
trainer.save_model(save_name)
config.save_pretrained(save_name)
tokenizer.save_pretrained(save_name)
