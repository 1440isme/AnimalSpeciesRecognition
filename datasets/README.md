# Datasets Folder

## Cấu trúc Dataset

Đặt các dataset của bạn vào thư mục này với cấu trúc sau:

```
datasets/
├── my_dataset_1/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class_n/
│       └── ...
├── my_dataset_2/
│   └── ...
```

## Yêu cầu

- Mỗi thư mục con trong dataset là một class
- Tên thư mục con = tên class (phải khớp với class_names trong model config)
- Định dạng ảnh hỗ trợ: `.jpg`, `.jpeg`, `.png`, `.webp`

## Animals-10 Classes

Các models trong hệ thống được train với Animals-10 dataset. Để benchmark chính xác, đảm bảo dataset của bạn có các class sau (viết thường):

- `butterfly`
- `cat`
- `chicken`
- `cow`
- `dog`
- `elephant`
- `horse`
- `sheep`
- `spider`
- `squirrel`

## Upload qua UI

Bạn cũng có thể upload dataset dạng ZIP file qua giao diện web:

1. Nén thư mục dataset thành file `.zip`
2. Truy cập trang Benchmark Studio
3. Kéo thả hoặc chọn file ZIP để upload

## Ví dụ tạo sample dataset

```bash
# Tạo cấu trúc dataset mẫu
mkdir -p animal_test/{cat,dog,elephant}

# Copy ảnh vào các thư mục tương ứng
cp /path/to/cat_images/*.jpg animal_test/cat/
cp /path/to/dog_images/*.jpg animal_test/dog/
cp /path/to/elephant_images/*.jpg animal_test/elephant/
```

