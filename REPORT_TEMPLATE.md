# CSC4005 Lab 7 Report – Compression: KD + Quantization Trade-offs

## 1. Thông tin

| STT | Họ tên | Mã sinh viên | Lớp |
|  1  | Trần Trường Giang    | 1671040009          | KHMT 16-01 |
|  2  | Nguyễn Văn Huy    | 1671040013          | KHMT 16-01 |

- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt1601_nhom11
- Kỹ thuật chọn: Quantization & Knowledge Distillation
- Link W&B nếu dùng KD: https://wandb.ai/giangtit1007-dainam-vietnam/csc4005-lab7-compression?nw=nwusergiangtit1007
- Link model nếu không commit trực tiếp: https://drive.google.com/drive/folders/1VY3JTBGc2LFmsMvQwqJwTsZxSI6mShxB?usp=sharing


# 2. Mô tả Baseline Model

| Nội dung                   | Giá trị                                                                 |
| -------------------------- | ----------------------------------------------------------------------- |
| Bài toán                   | Smart Campus Scene Classification                                       |
| Dataset                    | MIT Indoor Scenes 67 subset (5 classes)                                 |
| Số lớp                     | 5                                                                       |
| Baseline model             | Vision Transformer (ViT-B/16)                                           |
| Baseline format            | PyTorch / ONNX                                                          |
| Baseline checkpoint / ONNX | `checkpoints/teacher_vit_best_model.pt` / `models/vit_smartcampus.onnx` |
| Baseline model size        | 327.40 MB                                                               |

Baseline sử dụng mô hình Vision Transformer đã được huấn luyện ở Lab trước và export sang định dạng ONNX để phục vụ đánh giá và benchmark. Đây là mô hình có độ chính xác cao nhưng kích thước lớn và tốc độ suy luận còn hạn chế khi triển khai trên CPU hoặc thiết bị Edge.

---

# 3. Kỹ thuật nén đã chọn

Trong bài thực hành này, nhóm thực hiện **cả hai hướng Model Compression** gồm **Dynamic Quantization** và **Knowledge Distillation** để so sánh hiệu quả.

---

## 3.1 Dynamic Quantization

| Thông tin            | Giá trị                                    |
| -------------------- | ------------------------------------------ |
| Loại quantization    | Dynamic Quantization                       |
| Input model          | `models/vit_smartcampus.onnx`              |
| Output model         | `models/vit_smartcampus_dynamic_int8.onnx` |
| Dạng dữ liệu sau nén | INT8                                       |
| Công cụ              | `onnxruntime.quantization`                 |

### Mô tả

Dynamic Quantization chuyển trọng số mô hình từ FP32 sang INT8 trong quá trình suy luận nhằm giảm kích thước mô hình và tăng tốc inference trên CPU mà không cần huấn luyện lại mô hình.

---

## 3.2 Knowledge Distillation

| Thông tin     | Giá trị                       |
| ------------- | ----------------------------- |
| Teacher model | Vision Transformer (ViT-B/16) |
| Student model | MobileNetV2                   |
| alpha         | 0.5                           |
| temperature   | 4.0                           |
| epochs        | 10                            |
| batch size    | 16                            |
| optimizer     | AdamW                         |

### Công thức loss

```text
loss = alpha * CE(student_logits, labels)
      + (1 - alpha) * KD_loss(student_logits, teacher_logits, T)
```

Knowledge Distillation sử dụng mô hình ViT làm Teacher để truyền tri thức cho MobileNetV2 thông qua soft labels, giúp Student đạt hiệu năng cao nhưng có kích thước nhỏ và tốc độ suy luận nhanh hơn.

---

# 4. Kết quả đánh giá

## 4.1 So sánh Accuracy và Macro-F1

| Model          | Accuracy | Macro-F1 | Model size (MB) |
| -------------- | -------: | -------: | --------------: |
| Baseline ViT   |   98.10% |   97.48% |          327.40 |
| Quantized INT8 |   97.47% |   96.54% |           84.43 |
| KD Student     |   97.97% |   97.41% |            8.74 |

## Nhận xét

### Dynamic Quantization

* Accuracy giảm khoảng **0.63 điểm phần trăm**
* Macro-F1 giảm khoảng **0.94 điểm phần trăm**
* Kích thước mô hình giảm khoảng **74.2%**

Mức suy giảm Accuracy và Macro-F1 đều nhỏ hơn 1%, trong khi kích thước mô hình giảm rất mạnh, phù hợp triển khai trên CPU và hệ thống Smart Campus.

### Knowledge Distillation

Student MobileNetV2 đạt Accuracy **97.97%** và Macro-F1 **97.41%**, gần tương đương Teacher nhưng kích thước chỉ còn **8.74 MB**, chứng minh khả năng kế thừa tri thức hiệu quả từ Teacher.

---

# 5. Kết quả Benchmark

## 5.1 Baseline và Quantization

| Model     | Batch size | Mean latency (ms) | P95 latency (ms) | Throughput (img/s) | Size (MB) |
| --------- | ---------: | ----------------: | ---------------: | -----------------: | --------: |
| Baseline  |          1 |            243.97 |           331.85 |               4.10 |    327.40 |
| Quantized |          1 |            192.26 |           201.75 |               5.20 |     84.43 |
| Baseline  |          4 |           1246.73 |          1435.81 |               3.21 |    327.40 |
| Quantized |          4 |            804.97 |           912.06 |               4.97 |     84.43 |
| Baseline  |          8 |           2458.15 |          2758.35 |               3.25 |    327.40 |
| Quantized |          8 |           1715.23 |          1908.45 |               4.66 |     84.43 |

---

## 5.2 KD Student

| Batch size | Mean latency (ms) | P95 latency (ms) | Throughput (img/s) | Size (MB) |
| ---------: | ----------------: | ---------------: | -----------------: | --------: |
|          1 |             45.19 |            88.89 |              22.13 |      8.74 |
|          4 |            106.21 |           123.83 |              37.66 |      8.74 |
|          8 |            164.21 |           185.04 |              48.72 |      8.74 |

KD Student có tốc độ suy luận vượt trội so với ViT Baseline và Quantized, đặc biệt phù hợp cho các thiết bị Edge.

---

# 6. Bảng Trade-off

| Model          | Accuracy | Macro-F1 | Mean latency @bs=1 (ms) | Throughput @bs=1 | Size (MB) | Nhận xét                                                                  |
| -------------- | -------: | -------: | ----------------------: | ---------------: | --------: | ------------------------------------------------------------------------- |
| Baseline ViT   |   98.10% |   97.48% |                  243.97 |             4.10 |    327.40 | Accuracy cao nhưng mô hình lớn và suy luận chậm                           |
| Quantized INT8 |   97.47% |   96.54% |                  192.26 |             5.20 |     84.43 | Giảm khoảng 74% kích thước, tăng tốc CPU, Accuracy giảm nhẹ               |
| KD Student     |   97.97% |   97.41% |                   45.19 |            22.13 |      8.74 | Mô hình nhỏ nhất, throughput cao nhất và Accuracy gần tương đương Teacher |

---

# 7. Phân tích

## 1. Model sau nén nhỏ hơn bao nhiêu?

* Quantized INT8 giảm khoảng **74.2%**
* KD Student giảm khoảng **97.3%**

## 2. Latency thay đổi như thế nào?

* Quantized giảm khoảng **21%**
* KD Student giảm khoảng **81.5%**

## 3. Throughput thay đổi như thế nào?

* Quantized tăng khoảng **26.9%**
* KD Student tăng hơn **5 lần** so với Baseline

## 4. Accuracy và Macro-F1 giảm nhiều không?

Quantized chỉ giảm dưới **1%**, còn KD Student gần như giữ nguyên Accuracy và Macro-F1 của Teacher.

## 5. Nếu triển khai CPU hoặc Edge Device có chọn compressed model không?

Có.

* Với CPU và không muốn train lại: chọn Quantization.
* Với Edge Device hoặc IoT: chọn KD Student.

## 6. Nếu chỉ chọn một mô hình để triển khai

KD Student là lựa chọn tối ưu nhất vì đạt sự cân bằng tốt giữa Accuracy, Latency và Model Size.

---

# 8. Khi nào chọn KD, khi nào chọn Quantization?

| Quantization               | Knowledge Distillation     |
| -------------------------- | -------------------------- |
| Đã có model tốt            | Teacher quá lớn            |
| Không muốn train lại       | Sẵn sàng train Student     |
| Muốn giảm nhanh kích thước | Muốn mô hình cực nhỏ       |
| Tối ưu CPU inference       | Triển khai Edge Device     |
| Quy trình đơn giản         | Hiệu năng tổng thể tốt hơn |

Nếu thực hiện lại bài toán Smart Campus, nhóm ưu tiên **Knowledge Distillation** vì tạo ra mô hình MobileNetV2 chỉ **8.74 MB**, throughput trên **22 ảnh/giây** và Accuracy vẫn gần tương đương mô hình Teacher.

---

# 9. Kết luận

Trong bài thực hành này, nhóm đã triển khai thành công hai kỹ thuật Model Compression gồm Dynamic Quantization và Knowledge Distillation trên bài toán Smart Campus Scene Classification.

Dynamic Quantization giúp giảm khoảng **74% kích thước mô hình**, tăng tốc suy luận trên CPU trong khi Accuracy chỉ giảm dưới **1%**.

Knowledge Distillation tạo ra mô hình MobileNetV2 chỉ **8.74 MB**, giảm khoảng **97.3% dung lượng**, throughput tăng hơn **5 lần** nhưng Accuracy vẫn đạt **97.97%**, gần tương đương Teacher.

Kết quả thực nghiệm cho thấy việc nén mô hình mang lại hiệu quả rõ rệt về khả năng triển khai mà vẫn duy trì hiệu năng phân loại cao.

Qua phân tích trade-off giữa Accuracy, Latency và Model Size, nhóm nhận thấy **Knowledge Distillation là phương án tối ưu nhất cho hệ thống Smart Campus**, đặc biệt trong các môi trường triển khai trên CPU hoặc thiết bị Edge có tài nguyên hạn chế.
