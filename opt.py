import time

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *
import load_data
import pickle
import os
from Models import lenet5_relu
from Models import lenet5_tanh
from Models import vgg9_relu
from Models import vgg9_tanh


# def get_opt(model):
#     import torch
#     def skip(*args, **kwargs):
#         pass
#     torch.nn.init.kaiming_uniform_ = skip
#     torch.nn.init.uniform_ = skip
#     torch.nn.init.normal_ = skip
#     return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev, sparsity):

    print("Starting...")
    model.eval()
    model.to(dev)

    layers = find_layers(model)

    # ---- collect calibration data ----
    calib = []
    for batch in dataloader:
        calib.append(batch[0])
    calib = torch.cat(calib).to(dev)

    # print("Calibration data:", calib.shape)

    for name, layer in layers.items():

        # print(f"\nProcessing {name}")

        gpt = SparseGPT(layer)

        # ---- hook to capture layer IO during real forward ----
        def hook_fn(_, inp, out):
            gpt.add_batch(inp[0].data, out.data)

        handle = layer.register_forward_hook(hook_fn)

        # run full model forward calib.shape[0]
        for i in range(int(calib.shape[0])):
            model(calib[i:i+1])

        handle.remove()

        # ---- prune ----
        gpt.fasterprune(
            sparsity,
            percdamp=args.percdamp,
            blocksize=args.blocksize
        )
        gpt.free()

        torch.cuda.empty_cache()

    # print("Done.")

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache
    
    
    
@torch.no_grad()
def test(net_H, loader, device):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net_H.eval()

    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = net_H(data)

            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())

    return correct / len(loader.dataset)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, 
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'cifar10', 'cifar100', 'mnist'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=11, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity',
        type=float,
        nargs='+',
        default=[],
        help='Target sparsity list'
)
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )
    
    parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')
    
    parser.add_argument('--workers', type=int, default='4',
                        help='number of data loading workers (default: 4)')
    
    parser.add_argument('--model-class', type=str, default='default', choices=['default','lottery','tinyimagenet','imagenet'],
                        help='model class (default: default)')
    
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='load pretrained weights (default: False)')

    args = parser.parse_args()
    
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load_data.device(args.gpu)
    
    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load_data.dimension(args.dataset) 
    prune_loader, prune_set = load_data.dataloader(args.dataset, args.nsamples, True, args.workers, name = args.dataset + '_val')
    train_loader, train_set = load_data.dataloader(args.dataset, args.nsamples, True, args.workers, name = args.dataset + '_train')
    test_loader, test_set = load_data.dataloader(args.dataset, args.nsamples, False, args.workers, name = 'test')


    pretrained_path = 'Models/pkl/'
    ## load model ##
    # MNIST ReLU
    if args.model == "lenet5_relu":
        model = lenet5_relu.LeNet_relu()
        model_name = 'cnn_adv_relu.pth'
    # MNIST Tanh
    elif args.model == "lenet5_tanh":
        model = lenet5_tanh.LeNet_tanh()
        model_name =  'cnn_adv_tanh.pth'
    # CIFAR10 ReLU
    elif args.model == "vgg9_relu" and args.dataset == 'cifar10':
        model = vgg9_relu.VGG9_CIFAR10()
        model_name = 'vgg9_10_adv_relu_s2.pth'
    # CIFAR100 ReLU
    elif args.model == "vgg9_relu" and args.dataset == 'cifar100':
        model = vgg9_relu.VGG9_CIFAR10(num_classes=100)
        model_name = 'vgg9_100_wd_relu_s2.pth'
    # CIFAR10 Tanh   
    elif args.model == "vgg9_tanh" and args.dataset == 'cifar10':
        model = vgg9_tanh.VGG9_CIFAR10()
        model_name = 'vgg9_10_adv_tanh_s2.pth'
        
    model = model.to(device)
    pretrained_dict = torch.load(pretrained_path + model_name)
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model.eval()
    acc = test(model, test_loader, device)
    print(f'clean acc = {acc}')
    
    all_results = {
        "acc": [acc],
        "removed_num": [0],
    }

    if (args.sparsity or args.prunen) and not args.gmp:
        for sparsity in args.sparsity:
            tick = time.time()
            pretrained_dict = torch.load(pretrained_path + model_name)
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            opt_sequential(model, prune_loader, device, sparsity)
            
            # ---- sparsity stats ----
            total_params = 0
            total_zeros = 0
            layer_sp = {}
            for n, p in model.named_parameters():
                numel = p.numel()
                zeros = (p == 0).sum().item()

                layer_sp[n] = zeros / numel

                total_params += numel
                total_zeros += zeros
                # print(f'total = {numel}, removed = {zeros}')
                
            global_sp = total_zeros / total_params
            # print(time.time() - tick)
            
            acc = test(model, test_loader, device)
            print(f'acc = {acc}, removed p = {total_zeros}')

            # ---- store ----
            # all_results["sparsity"].append(sparsity)
            # all_results["global_sparsity"].append(global_sp)
            if sparsity == 1:
                total_zeros = total_params
                
            all_results["acc"].append(acc)
            all_results["removed_num"].append(total_zeros)
            # all_results["layer_sparsity"].append(layer_sp)
            
    os.makedirs("Results", exist_ok=True)

    save_path = f"Results/{args.model}gpt_results.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Saved variables to {save_path}")

    if args.save:
        model.save_pretrained(args.save)
