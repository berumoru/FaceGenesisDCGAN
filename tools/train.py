import cv2
import os, sys
import torch
import random
import argparse
from tqdm import tqdm
from torch import optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
# 関数をインポート
from models import *
from utils import *
from shared_vars import log_loss_G, log_loss_D

def main(args):
    # ここで引数を使用した処理を行います
    print("========== config ==========")
    print(f"zip_file_path: {args.zip_file_path}")
    print(f"extract_path: {args.extract_path}")
    print(f"result_dir: {args.result_dir}")
    print(f"haar_file: {args.haar_file}")
    print(f"image_size: {args.image_size}")
    print(f"batch_size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"nz: {args.nz}")

    # 主要処理
    dataset_path = os.path.dirname(args.extract_path) # 親フォルダ
    clean_folder(args.extract_path) #フォルダ初期化
    jpg_files = extract_jpg_files(args.zip_file_path, args.extract_path)
    cascade = cv2.CascadeClassifier(args.haar_file)
    print("========== load dataset ===========")
    resized_faces = detect_resize_and_rename_faces(jpg_files, cascade, image_size=args.image_size, output_path=args.extract_path)
    # face_cut 50枚のサンプルを保存
    selected_faces = random.sample(resized_faces, min(50, len(resized_faces)))
    combined_image = combine_images(selected_faces, image_size=args.image_size)
    try:
        # ディレクトリを作成し、すでに存在する場合は例外を発生させる
        os.makedirs(args.result_dir, exist_ok=False)
    except FileExistsError:
        # 例外をキャッチし、エラーメッセージを表示後にプログラムを終了
        print(f"Error: The directory '{args.result_dir}' already exists.")
        sys.exit(1)
    combined_image.save(f'{args.result_dir}/combined_faces.jpg')
    # サブフォルダのみを削除
    clean_subfolders(args.extract_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 1. データの読み込み ===
    # datasetの準備
    # dataset = datasets.ImageFolder(dataset_path,
    #     transform=transforms.Compose([
    #         transforms.ToTensor()
    # ]))

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # ランダムな水平フリップ
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] に正規化
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    # dataloaderの準備
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model_G = Generator().to(device)
    model_D = Discriminator().to(device)

    #重み初期化
    model_G.apply(weights_init)
    model_D.apply(weights_init)

    params_G = optim.Adam(model_G.parameters(),
        lr=0.0002, betas=(0.5, 0.999))
    params_D = optim.Adam(model_D.parameters(),
        lr=0.0002, betas=(0.5, 0.999))

    # 途中結果の確認用の潜在特徴z
    check_z = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
    print("========== train start ===========")
    for epoch in tqdm(range(args.epochs), desc=" Train Processing"):
        train_dcgan(model_G, model_D, params_G, params_D, data_loader, args.batch_size, device=device, nz=args.nz)
        # 10エポックごとにモデルと生成画像を保存
        if epoch % 10 == 0:
            # ディレクトリがない場合は作成
            os.makedirs(f"{args.result_dir}/Weight_Generator", exist_ok=True)
            os.makedirs(f"{args.result_dir}/Weight_Discriminator", exist_ok=True)
            os.makedirs(f"{args.result_dir}/Generated_Image", exist_ok=True)
            # モデルの保存
            torch.save(
                model_G.state_dict(),
                f"{args.result_dir}/Weight_Generator/G_{epoch}.pth",
                pickle_protocol=4)
            torch.save(
                model_D.state_dict(),
                f"{args.result_dir}/Weight_Discriminator/D_{epoch}.pth",
                pickle_protocol=4)
            # 生成画像の保存
            generated_img = model_G(check_z)
            save_image(generated_img,
                    f"{args.result_dir}/Generated_Image/{epoch}.jpg")
    plot_and_save_loss(log_loss_G, log_loss_D, args.result_dir)
    print("========== Done !! ==========")

if __name__ == "__main__":
    # コマンドライン引数を解析するためのパーサーを作成
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_file_path", type=str, help="zipファイルのパス")
    parser.add_argument("--extract_path", type=str, help="展開するフォルダのパス")
    parser.add_argument("--result_dir", type=str, help="結果を保存するディレクトリ")
    parser.add_argument("--haar_file", type=str, default="./tools/haarcascade_frontalface_default.xml", help="Haar Cascadeのファイルパス")
    parser.add_argument("--image_size", type=int, nargs=2, default=(128, 128), help="画像サイズ")
    parser.add_argument("--batch_size", type=int, default=128, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=500, help="エポック数")
    parser.add_argument("--nz", type=int, default=100, help="潜在特徴の次元数")
    args = parser.parse_args()
    main(args)