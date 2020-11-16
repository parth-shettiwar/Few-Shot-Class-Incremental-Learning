# Few Shot Class Incremental Learning based on GANs  
Solves the problem of catastrophic forgetting in GANs using the idea of generating pseudo prototypes.  
Tested on mini-Imagenet as of now.  
**Training**  
run python train.py -data imagenet_sub -epochs_gan 10000 -tradeoff 1 -epochs 101 -lr_decay_step 100 -log_dir imagenet_sub_10tasks -dir /dataset -gpu 0  
**Testing**  
python test.py -data imagenet_sub -num_task 1 -epochs 101 -dir /dataset -gpu 0 -r checkpoints/imagenet_sub_10 -name imagenet_sub_10
Download the mini-imagenet in a "dataset" folder





