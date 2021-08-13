### Command
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --data /ssd1/chenwy/ \
--dataset cifar10|cifar100|svhn \
--arch resnet18|resnet50|resnet101 \
--exp_name \
--inference --checkpoint
```