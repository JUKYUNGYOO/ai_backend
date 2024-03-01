#! /bin/bash

# Train with A100 x 8 : before
# Train with TU102 x 3

python detrex/tools/train_net.py \
  --config-file orion/config/train.py \
  --num-gpus 3 \
  train.amp.enabled=False \
  train.init_checkpoint=/workspace/output/model/dino_swin_base_384_4scale_12ep.pth \
  train.output_dir=output/test_model/ 



#  True : RuntimeError: CUDA out of memory. Tried to allocate 426.00 MiB (GPU 1; 23.65 GiB total capacity; 21.88 GiB already allocated; 293.19 MiB free; 22.29 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
# False : RuntimeError: CUDA out of memory. Tried to allocate 958.00 MiB (GPU 1; 23.65 GiB total capacity; 21.35 GiB already allocated; 463.19 MiB free; 22.12 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF