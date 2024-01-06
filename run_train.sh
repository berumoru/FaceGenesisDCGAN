python3 ./tools/train.py \
--zip_file_path "./dataset/picture.zip" \
--extract_path "./dataset/extracted_pictures" \
--result_dir "Result/try3" \
--haar_file "./tools/haarcascade_frontalface_default.xml" \
--image_size 128 128  \
--batch_size 128 \
--epochs 500 \
--nz 100