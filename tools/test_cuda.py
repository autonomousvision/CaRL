import torch
import os
import datetime

rank = int(os.environ['RANK'])  # Rank across all processes
world_size = int(os.environ['WORLD_SIZE'])  # Number of processes
model = torch.nn.Linear(10, 10)

device = f'cuda:{rank}'

torch.distributed.init_process_group(backend='nccl',
                                     store=None,
                                     world_size=world_size,
                                     rank=rank,
                                     timeout=datetime.timedelta(minutes=45))
model.to(device)

model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=None,
                                                  output_device=None,
                                                  broadcast_buffers=False,
                                                  find_unused_parameters=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)



input = torch.randn(10).to(device)
label = torch.ones(10).to(device)


print('Start training')
for i in range(100):
  optimizer.zero_grad()
  pred = model(input)
  loss = torch.nn.functional.l1_loss(pred, label)
  print(f'Rank: {rank} Loss: ', loss)
  loss.backward()
  optimizer.step()

print('Finished training')
