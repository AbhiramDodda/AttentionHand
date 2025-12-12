import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.hack import disable_verbosity
disable_verbosity()

# Configs
resume_path = 'weights/sd15_ini.ckpt'
batch_size = 1
logger_freq = 100
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# 1. Load Model (CPU-only first, for flexibility)
# The model will be automatically moved to the correct device by the Trainer
model = create_model('models/attentionhand.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# 2. Setup DataLoaders
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)


# 3. Setup Trainer
# The original 'gpus=1' is deprecated/changed. We now use 'accelerator' and 'devices'.
# - accelerator='gpu': Specifies to use GPU hardware.
# - devices=1: Specifies to use 1 device (GPU).
# - precision='32-true': Modern name for precision=32.
trainer = pl.Trainer(
    accelerator='gpu',      # Specify the hardware type (e.g., 'gpu', 'cpu', 'tpu')
    devices=1,              # Specify the number of devices to use (e.g., 1, [0, 1], 'auto')
    precision='32-true',    # Use '32-true' for standard float32 training
    callbacks=[logger]
)


# 4. Train!
trainer.fit(model, dataloader)