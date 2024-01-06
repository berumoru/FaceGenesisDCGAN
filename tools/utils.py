import os
import cv2
import shutil
import zipfile
import torch
from tqdm import tqdm
import random
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from statistics import mean
from shared_vars import log_loss_G, log_loss_D

def clean_folder(folder_path):
    if os.path.exists(folder_path):
        # フォルダ内のすべてのファイルとサブフォルダを削除
        shutil.rmtree(folder_path)
    # クリーンなフォルダを作成
    os.makedirs(folder_path, exist_ok=True)
    
def clean_subfolders(folder_path):
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                 
def extract_jpg_files(zip_file_path, extract_path='./extracted_pictures'):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        jpg_files = [os.path.join(extract_path, f) for f in zip_ref.namelist() if f.endswith('.jpg')]
        return jpg_files

def detect_resize_and_rename_faces(file_paths, cascade, image_size=(128, 128), output_path='./extracted_pictures'):
    os.makedirs(output_path, exist_ok=True)
    resized_faces = []
    saved_images_count = 0  # ここで変数を初期化
    for file_path in tqdm(file_paths, desc=" Data Processing"):
        img = cv2.imread(file_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            img_gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)  # 最小の顔サイズを30x30ピクセルに設定
        )
        if len(faces) == 0:
            os.remove(file_path)
            continue
        margin_ratio = 0.1
        for x, y, w, h in faces:
            # マージンを計算
            margin_w = int(w * margin_ratio)
            margin_h = int(h * margin_ratio)
            # 余裕を持たせた座標を計算
            x_start = max(x - margin_w, 0)
            y_start = max(y - margin_h, 0)
            x_end = min(x + w + margin_w, img.shape[1])
            y_end = min(y + h + margin_h, img.shape[0])
            # 余裕を持たせた部分を切り取り
            face_cut = img[y_start:y_end, x_start:x_end]
            face_resized = cv2.resize(face_cut, image_size)
            resized_faces.append(face_resized)
            # リサイズされた画像を新しい名前で保存
            saved_images_count += 1
            new_file_name = f"{saved_images_count}.jpg"
            cv2.imwrite(os.path.join(output_path, new_file_name), face_resized)
    return resized_faces

def combine_images(images, grid_size=(10, 5), image_size=(128, 128)):
    if len(images) > 50:
        images = random.sample(images, 30)  # ランダムに30枚選択
    grid_img = Image.new('RGB', (grid_size[0] * image_size[0], grid_size[1] * image_size[1]))
    for i, img in enumerate(images):
        row = i // grid_size[0]
        col = i % grid_size[0]
        grid_img.paste(Image.fromarray(img), (col * image_size[0], row * image_size[1]))
    return grid_img

# 訓練関数
def train_dcgan(model_G, model_D, params_G, params_D, data_loader, batch_size, device, nz):
    # ロスを計算するときのラベル変数
    ones = torch.ones(batch_size).to(device) # 正例 1
    zeros = torch.zeros(batch_size).to(device) # 負例 0
    loss_f = nn.BCEWithLogitsLoss()
    for real_img, _ in data_loader:
        batch_len = len(real_img)

        # == Generatorの訓練 ==
        # 偽画像を生成
        z = torch.randn(batch_len, nz, 1, 1).to(device)
        fake_img = model_G(z)

        # 偽画像の値を一時的に保存 => 注(１)
        fake_img_tensor = fake_img.detach()

        # 偽画像を実画像（ラベル１）と騙せるようにロスを計算
        out = model_D(fake_img)
        loss_G = loss_f(out, ones[: batch_len])
        log_loss_G.append(loss_G.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_G.backward()
        params_G.step()
        # == Discriminatorの訓練 ==
        # sample_dataの実画像
        real_img = real_img.to(device)

        # 実画像を実画像（ラベル１）と識別できるようにロスを計算
        real_out = model_D(real_img)
        loss_D_real = loss_f(real_out, ones[: batch_len])

        # 計算省略 => 注（１）
        fake_img = fake_img_tensor

        # 偽画像を偽画像（ラベル０）と識別できるようにロスを計算
        fake_out = model_D(fake_img_tensor)
        loss_D_fake = loss_f(fake_out, zeros[: batch_len])

        # 実画像と偽画像のロスを合計
        loss_D = loss_D_real + loss_D_fake
        log_loss_D.append(loss_D.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_D.backward()
        params_D.step()

    return mean(log_loss_G), mean(log_loss_D)

def plot_and_save_loss(log_loss_G, log_loss_D, result_dir, file_name="DCGAN_Loss.png"):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(log_loss_G, label="G")
    plt.plot(log_loss_D, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # 画像を保存
    plt.savefig(f"{result_dir}/{file_name}")