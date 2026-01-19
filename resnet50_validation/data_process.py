import numpy as np
import onnxruntime as ort
import time
import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--preprocess", action="store_true", help="run preprocess")
parser.add_argument("--postprocess", action="store_true", help="run postprocess")

args = parser.parse_args()

num_tests = 10

def load_and_preprocess_images(image_folder,save_folder="input"):
    data_list = []
    img_names = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    
    for idx,name in enumerate(img_names):
        path = os.path.join(image_folder, name)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: failed to load {path}")
            continue
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # add batch dim
        img = np.expand_dims(img, axis=0)
        save_path = os.path.join(save_folder, f"input_{idx}.npy")
        np.save(save_path, img)
        data_list.append(img)
    return data_list

input_folder="input"
if args.preprocess:
    # Create input directory if it doesn't exist
    os.makedirs(input_folder, exist_ok=True)
    image_folder = "val_data"
    data_list = load_and_preprocess_images(image_folder,input_folder)
    print(f"Loaded {len(data_list)} images, shape[0]: {data_list[0].shape}")

def compare_output(output_dir1, output_dir2):
    files1 = sorted([f for f in os.listdir(output_dir1) if f.endswith(".npy")])
    files2 = sorted([f for f in os.listdir(output_dir2) if f.endswith(".npy")])
    for f1, f2 in zip(files1, files2):
        path1 = os.path.join(output_dir1, f1)
        path2 = os.path.join(output_dir2, f2)
        data1 = np.load(path1)
        data2 = np.load(path2)
        max_diff = np.max(np.abs(data1 - data2))
        golden_max = np.max(np.abs(data1))
        max_diff_percent = (max_diff / golden_max) * 100 if golden_max != 0 else float('inf')
        print(f"\t{path1} vs {path2} -> max_diff: {max_diff:.6f}, max_diff_percent: {max_diff_percent:.4f}%")

if args.postprocess :
    if os.path.isfile("output_cpu/output_0_0.npy") and os.path.isfile("output_vek385/output_0_0.npy"):
        print("\nCPU vs VEK385:")
        compare_output("output_cpu","output_vek385")

    if os.path.isfile("output_cpu/output_0_0.npy") and os.path.isfile("output_ryzen/output_0_0.npy"):
        print("\nCPU vs Ryzen:")
        compare_output("output_cpu","output_ryzen")
        
    if os.path.isfile("output_vek385/output_0_0.npy") and os.path.isfile("output_ryzen/output_0_0.npy"):
        print("\nVEK385 vs Ryzen:")
        compare_output("output_vek385","output_ryzen")