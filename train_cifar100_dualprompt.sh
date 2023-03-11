python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        cifar100_dualprompt \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --data-path ./cifar_dataset \
        --output_dir ./output --seed 7085 --epochs 5 > 'exp7.txt'