import os
# att_list = ["CoordAttention","SimAM", "cbam"]
# att_list = ["CoordAttention"]
# exp4 = "python seg_train.py  --doubleEncoder 1  --freeze 0 --resolution 512 --train_batch_size 2 --backbone timm-efficientnet-b5 --encoder_weight E5 --fuse_method gated "

# python seg_train.py  --doubleEncoder 0  --freeze 0 --resolution 512 --train_batch_size 4 --backbone timm-efficientnet-b5
# python seg_train.py  --doubleEncoder 0  --freeze 0 --resolution 512 --train_batch_size 4 --backbone timm-efficientnet-b5 --encoder_weight E5 --att_method ACmix
# for att in att_list:
#     cmd = exp4+f"--att_method {att}"
#     print(f"Running: {cmd}")
    # os.system(cmd)
    
cmd1 = "python seg_train.py  --doubleEncoder 0  --freeze 0 --resolution 512 --train_batch_size 4 --backbone timm-efficientnet-b5 --num_train_epochs 250"
# cmd2 = "python seg_train.py  --doubleEncoder 1  --freeze 0 --resolution 512 --train_batch_size 4 --backbone timm-efficientnet-b5 --encoder_weight E5 --fuse_method gated windowcross"
# cmd3 = "python seg_train.py  --doubleEncoder 1  --freeze 0 --resolution 512 --train_batch_size 4 --backbone timm-efficientnet-b5 --encoder_weight E5 --fuse_method gated"

os.system(cmd1)
# os.system(cmd2)