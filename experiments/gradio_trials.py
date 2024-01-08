from transformers import BertModel, PreTrainedModel, AutoConfig, AutoTokenizer
from torch import nn
import gradio as gr
class CustomBERTModel(PreTrainedModel):
      def __init__(self, config, transformer_model_name_or_path, num_feats):
            super(CustomBERTModel, self).__init__(config)
            self.bert = BertModel.from_pretrained(
                  transformer_model_name_or_path,
                  config=config
            )
            ### New layers:
            self.regression_layer = nn.Sequential(
                  nn.Linear(config.hidden_size, 384),
                  nn.LeakyReLU(),
                  nn.Linear(384, 192),
                  nn.LeakyReLU(),
                  nn.Linear(192, 96),
                  nn.LeakyReLU(),
                  nn.Linear(96, 48),
                  nn.LeakyReLU(),
                  nn.Linear(48, 24),
                  nn.LeakyReLU(),
                  nn.Linear(24, 12),
                  nn.LeakyReLU(),
                  nn.Linear(12, num_feats),
                  nn.Sigmoid(),
            )

      def forward(self, **inputs):
            bert_outputs = self.bert(**inputs)
            logits = self.regression_layer(bert_outputs.pooler_output)
            return logits

      def _init_weights(self, module):
            self.bert._init_weights(module)


class Interface:
      def __init__(self) -> None:
            base_model = 'bert-base-uncased'
            checkpoint = '/home/yuuhanase/FPTU/EXE101/PaperClipAI_EnglishGrading/EnglishGradingModel'
            config = AutoConfig.from_pretrained(base_model)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = CustomBERTModel.from_pretrained(
                  checkpoint, 
                  config=config, 
                  transformer_model_name_or_path=base_model,
                  num_feats=6
            )
      
      def grading(self, text):
            tokenized_input = self.tokenizer(text, return_tensors='pt', truncation=True)#.to(torch.device("cuda"))
            output = self.model(**tokenized_input)[0]
            feats = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
            result = ''
            for i in range(6):
                  result += f'{feats[i]}: {output[i] * 5.0}\n'
            return result

      def run(self):
            demo = gr.Interface(fn=self.grading, inputs="textbox", outputs="textbox")
            demo.launch(share=True)

if __name__ == '__main__':
      interface = Interface()
      interface.run()