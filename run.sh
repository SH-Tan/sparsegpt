# python opt.py --model lenet5_tanh --dataset mnist --sparsity 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
# python opt.py --model vgg9_tanh --dataset cifar10 --sparsity 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 --gpu 1
python opt.py --model vgg9_relu --dataset cifar100 --sparsity 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1