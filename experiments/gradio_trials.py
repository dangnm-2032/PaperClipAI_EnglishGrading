from transformers import T5EncoderModel, PreTrainedModel, AutoConfig, AutoTokenizer
from torch import nn
import gradio as gr
import torch

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
            t5_outputs = self.t5(
                  input_ids=inputs['input_ids'],
                  attention_mask=inputs['attention_mask'])
            logits = self.regression_layer(t5_outputs.last_hidden_state)
            return logits

      def _init_weights(self, module):
            self.t5._init_weights(module)


class Interface:
      def __init__(self) -> None:
            base_model = 'google/flan-t5-large'
            checkpoint = '/home/yuuhanase/FPTU/EXE101/PaperClipAI_EnglishGrading/artifacts/trained_model/EnglishGrading_t5_regression_5e'
            config = AutoConfig.from_pretrained(checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = CustomT5Model.from_pretrained(
                  checkpoint, 
                  config=config, 
                  base_model=base_model,
            )
      
      def inference(self, inp_text):
            tokenized_input = self.tokenizer(
                  inp_text, 
                  return_tensors='pt',
                  max_length=1548,
                  padding='max_length').to(self.model.device)
            output = self.model(**tokenized_input)[0]
            output
            feats = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
            result = {}
            for i in range(6):
                  result[feats[i]] = round((output[i].item()*5.0)*2)/2.0
            tokenized_input = tokenized_input.to('cpu')
            output = output.to('cpu')

            del tokenized_input, output
            torch.cuda.empty_cache()

            return result

      def grading(self, text):
            result = self.inference(text)
            output = ''
            for criteria in result:
                  output += f'{criteria}: {result[criteria]}\n'
            return output

      def run(self):
            demo = gr.Interface(fn=self.grading, inputs="textbox", outputs="textbox")
            demo.launch(share=True)

if __name__ == '__main__':
      interface = Interface()
      interface.run()