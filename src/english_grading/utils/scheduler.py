# ✨ Import essential tools for scheduling magic ✨
from transformers import (
   get_linear_schedule_with_warmup,  #  Linear progression with a gentle warmup 🪜
   get_cosine_schedule_with_warmup,  #  Cyclic waves for rhythmic learning 
   get_polynomial_decay_schedule_with_warmup,  #  Gradual decay for a smooth landing 
   get_constant_schedule_with_warmup  #  Steady as she goes, no surprises here 
)

def get_scheduler(optimizer, config, num_train_steps):
   """
   Selects the most suitable scheduler for the task at hand,
   tailoring the learning rate to optimize training progress. 

   Args:
       optimizer (torch.optim.Optimizer): The optimizer to be guided by the scheduler. 
       config (Config): Configuration object containing scheduler preferences. ⚙️
       num_train_steps (int): The total number of training steps planned. ️

   Returns:
       torch.optim.lr_scheduler.LambdaLR: The chosen scheduler, ready to orchestrate learning. 
   """

   #  Discerning the most fitting scheduler 
   if config.scheduler.scheduler_type == 'constant_schedule_with_warmup':
       scheduler = get_constant_schedule_with_warmup(
           optimizer,
           num_warmup_steps=config.scheduler.constant_schedule_with_warmup.n_warmup_steps  # 🪜 Warm up for a smooth start
       )
   elif config.scheduler.scheduler_type == 'linear_schedule_with_warmup':
       scheduler = get_linear_schedule_with_warmup(
           optimizer,
           num_warmup_steps=config.scheduler.linear_schedule_with_warmup.n_warmup_steps,  # 🪜 Warm up, then climb steadily 
           num_training_steps=num_train_steps  #  Finish strong at the end
       )
   elif config.scheduler.scheduler_type == 'cosine_schedule_with_warmup':
       scheduler = get_cosine_schedule_with_warmup(
           optimizer,
           num_warmup_steps=config.scheduler.cosine_schedule_with_warmup.n_warmup_steps,  # 🪜 Warm up for rhythmic waves 
           num_cycles=config.scheduler.cosine_schedule_with_warmup.n_cycles,  #  Ride the learning waves 
           num_training_steps=num_train_steps  #  Anchor at the finish line
       )
   elif config.scheduler.scheduler_type == 'polynomial_decay_schedule_with_warmup':
       scheduler = get_polynomial_decay_schedule_with_warmup(
           optimizer,
           num_warmup_steps=config.scheduler.polynomial_decay_schedule_with_warmup.n_warmup_steps,  # 🪜 Warm up, then gracefully descend 
           num_training_steps=num_train_steps,  #  Land softly at the end
           power=config.scheduler.polynomial_decay_schedule_with_warmup.power,  #  Control the decay curve 
           lr_end=config.scheduler.polynomial_decay_schedule_with_warmup.min_lr  #  Set the final landing point 
       )
   else:
       raise ValueError(f'Unknown scheduler: {config.scheduler.scheduler_type}')  #  Sound the alarm for unknown schedulers 

   return scheduler  # ✨ Return the chosen scheduler to guide the learning journey ✨