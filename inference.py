# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime

from export_models import load_model

CIFAR10_CLASSES = [
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

NORMALIZATION = {
    'cifar10': {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
    'cifar100': {'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761)},
}


def get_transform(img_size: int, dataset: str):
    if dataset not in NORMALIZATION:
        raise ValueError(f"Unsupported dataset: {dataset}")

    norm = NORMALIZATION[dataset]
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm['mean'], std=norm['std']),
    ])


def load_torchscript(path: str, device: str):
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model


def load_onnx(path: str, device: str):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(path, providers=providers)
    return session


def infer_onnx(session: onnxruntime.InferenceSession, input_tensor: torch.Tensor, device: str):
    input_np = input_tensor.cpu().numpy()
    ort_inputs = {session.get_inputs()[0].name: input_np}
    ort_outs = session.run(None, ort_inputs)
    output = torch.from_numpy(ort_outs[0])
    return output


def infer(model: torch.nn.Module, input_tensor: torch.Tensor, device: str):
    model.to(device)
    input_tensor = input_tensor.to(device).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    return output


def main():
    parser = argparse.ArgumentParser(description='Simple inference for CIFAR model')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--checkpoint', type=str, default=None, help='PyTorch checkpoint path (.t7)')
    parser.add_argument('--torchscript', type=str, default=None, help='TorchScript model path (.pt)')
    parser.add_argument('--onnx', type=str, default=None, help='ONNX model path (.onnx)')
    parser.add_argument('--model_type', type=str, default='vit', choices=['vit', 'cait', 'swin'], help='Model architecture type')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset normalization')
    parser.add_argument('--img_size', type=int, default=32, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')

    args = parser.parse_args()

    provided = [args.checkpoint is not None, args.torchscript is not None, args.onnx is not None]
    if sum(provided) != 1:
        raise ValueError('Provide exactly one of --checkpoint, --torchscript, or --onnx.')

    device = args.device

    if args.torchscript is not None:
        model = load_torchscript(args.torchscript, device)
        onnx_session = None
    elif args.onnx is not None:
        onnx_session = load_onnx(args.onnx, device)
        model = None
    else:
        model = load_model(args.checkpoint, args.model_type, device=device)
        onnx_session = None

    transform = get_transform(args.img_size, args.dataset)
    image = Image.open(args.image).convert('RGB')
    input_tensor = transform(image)

    if onnx_session is not None:
        output = infer_onnx(onnx_session, input_tensor, device)
    else:
        output = infer(model, input_tensor, device)

    probabilities = torch.softmax(output, dim=1)[0]
    top_prob, top_idx = torch.max(probabilities, dim=0)

    class_name = None
    if args.dataset == 'cifar10':
        class_name = CIFAR10_CLASSES[top_idx.item()]

    print('Prediction:')
    print(f'  class_index: {top_idx.item()}')
    if class_name is not None:
        print(f'  class_name: {class_name}')
    print(f'  confidence: {top_prob.item():.4f}')


if __name__ == '__main__':
    main()
