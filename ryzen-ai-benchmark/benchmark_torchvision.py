import argparse
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

from tqdm import tqdm
import torch
import torchvision

def main(args):
    device = 'cpu'

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    
    if args.model == "resnet50":
        model = torchvision.models.resnet50()
    elif args.model == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2()
    elif args.model == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2()
    else:
        raise ValueError(f"Invalid model name : {args.model}")

    model.eval()
    model.to(device)

    print("Input shape :", args.input_shape)
    input = torch.rand(args.input_shape).to(device)

    with torch.no_grad():
        # Warm up
        model(input)
        
        if args.parallel < 2:
            for _ in tqdm(range(args.test_num), desc="Processing batches"):
                model(input)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                queue = Queue(maxsize=args.parallel-1)

                def get_results():
                    for _ in tqdm(range(args.test_num), desc="Processing batches"):
                        future = queue.get()
                        future.result()
                
                thread = Thread(target=get_results)
                thread.start()

                for _ in range(args.test_num):
                    queue.put(executor.submit(model, input))

                thread.join()
    
    if args.export_onnx is not None:
        torch.onnx.export(
            model,
            input,
            args.export_onnx,
            export_params=True,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', type=str)
    parser.add_argument('--cpu_threads', type=int, default=0)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=40)
    parser.add_argument('--input_shape', type=int, nargs="+", default=[1, 3, 224, 224])
    parser.add_argument('--export_onnx', type=str, default=None)
    args = parser.parse_args()

    assert args.test_num > 0

    main(args)
