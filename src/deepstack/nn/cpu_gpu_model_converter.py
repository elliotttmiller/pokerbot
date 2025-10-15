"""
CPU/GPU model converter for DeepStack neural network pipeline.
Ported from DeepStack Lua cpu_gpu_model_converter.lua.
"""
import torch

def convert_gpu_to_cpu(gpu_model_path):
    info = torch.load(gpu_model_path + '_gpu.info')
    info['gpu'] = False
    model = torch.load(gpu_model_path + '_gpu.model')
    model = model.float()
    torch.save(gpu_model_path + '_cpu.info', info)
    torch.save(gpu_model_path + '_cpu.model', model)

def convert_cpu_to_gpu(cpu_model_path):
    info = torch.load(cpu_model_path + '_cpu.info')
    info['gpu'] = True
    model = torch.load(cpu_model_path + '_cpu.model')
    model = model.cuda()
    torch.save(cpu_model_path + '_gpu.info', info)
    torch.save(cpu_model_path + '_gpu.model', model)

def convert_model(model_path, to='cpu'):
    """Convert model between CPU and GPU formats."""
    if to == 'cpu':
        return convert_gpu_to_cpu(model_path)
    else:
        return convert_cpu_to_gpu(model_path)
