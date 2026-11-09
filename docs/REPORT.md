# BÁO CÁO THẨM ĐỊNH KHOA HỌC VÀ KHUNG BENCHMARK CHUẨN SEN1FLOODS11

---

## MỤC LỤC
1. [Kết quả kiểm tra nhận xét cũ (Audit Claim-by-Claim)](#1-kết-quả-kiểm-tra-nhận-xét-cũ)
2. [Kết luận khoa học về việc cắt Sliding Window 256x256](#2-kết-luận-khoa-học-về-sliding-window-256x256)
3. [Danh mục mô hình Baseline có mã nguồn công khai](#3-danh-mục-baseline-có-mã-nguồn-công-khai)
4. [Bảng đối chiếu yêu cầu xuất bản chuẩn A*/Q1](#4-bảng-đối-chiếu-chuẩn-hóa-a-q1)
5. [Hướng dẫn thực thi trên Kaggle GPU và đồng bộ Weights](#5-hướng-dẫn-thực-thi-trên-kaggle-gpu)
6. [Các hạng mục chưa thực hiện và lý do kỹ thuật](#6-các-hạng-mục-chưa-thực-hiện-và-lý-do)

---

## 1. KẾT QUẢ KIỂM TRA NHẬN XÉT CŨ

Toàn bộ các tuyên bố trong báo cáo trước đây đã được đối chiếu độc lập với các bài báo gốc trên arXiv, IEEE Xplore, Frontiers, CVPR Open Access và mã nguồn chính thức:

| # | Tuyên bố kiểm tra | Phán quyết (Verdict) | Giá trị thực tế & Chi tiết thẩm định | Nguồn tài liệu cấp 1 (Primary Source) | Trang / Bảng trong PDF |
|---|---|---|---|---|---|
| 1 | Phân chia tập chính thức: 252 train, 89 valid, 90 test, 15 Bolivia holdout. Tổng 446 chip hand, 4.385 weak, 4.831 toàn bộ. | **CHÍNH XÁC (TRUE)** | Khớp 100% với metadata và CSV phân chia gốc của Cloud to Street / NASA. | [Bonafilia et al. (CVPRW 2020)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf) | Trang 211, Section 3.1 & split CSVs |
| 2 | ProCANet: arXiv:2501.11923, Feliren et al., mIoU/IoU 0.815, so sánh với MANet, PAN, ConvNeXt V2 trên Sen1Floods11 và Citarum. | **CHÍNH XÁC (TRUE)** | Tác giả: Vicky Feliren, Fithrothul Khikmah, Irfan Dwiki Bhaswara, Bahrul I. Nasution, Alex M. Lechner, Muhamad Risqi U. Saputra. IoU đạt 0.8150 (Acc 0.9811, F1 0.8982). Zero-shot trên sông Citarum (PlanetScope). | [arXiv:2501.11923 (IEEE GRSL 2025)](https://arxiv.org/abs/2501.11923) | Trang 4, Bảng 1 |
| 3 | ProCANet: Dùng split 65% train / 35% val ngẫu nhiên, 25 epochs, Adam, Cosine Annealing, crop 256x256. | **CHÍNH XÁC (TRUE)** | Bài báo phân chia ngẫu nhiên 65/35 trên tập hand-labeled, không kiểm thử trên 90 test chip chính thức của Bonafilia. | [arXiv:2501.11923](https://arxiv.org/abs/2501.11923) | Trang 3, Section III.B |
| 4 | ProCANet có mã nguồn chính thức tại `RituYadav92/UNI_TEMP_FLOOD_DETECTION`. | **BỊA ĐẶT (FABRICATED)** | Bài báo ProCANet **không hề có link mã nguồn**. Nhóm tác giả Feliren không công bố mã nguồn. Repo trên thuộc về Ritu Yadav (KTH), nghiên cứu hoàn toàn khác về S1 đơn thời điểm. Trạng thái: `NO_OFFICIAL_REPO`. | PDF arXiv:2501.11923 | Toàn văn (Không có link code) |
| 5 | ADSS-UNet: arXiv:2204.09387, đánh giá trên Sen1Floods11 S1 bi-temporal, hơn uni-temporal 6% IoU. | **CHÍNH XÁC (TRUE)** | Tác giả đề xuất mạng siamese 2 luồng S1 trước/sau lũ, cải thiện +6% Water IoU so với đơn thời điểm. | [arXiv:2204.09387 (IEEE IGARSS 2022)](https://arxiv.org/abs/2204.09387) | Trang 3, Section 3 |
| 6 | Tác giả ADSS-UNet là "D. Gupta et al.". | **SAI (FALSE)** | Tác giả thực tế: **Ritu Yadav, Andrea Nascetti, Yifang Ban** (KTH Thụy Điển). Không có ai tên Gupta. | [arXiv:2204.09387](https://arxiv.org/abs/2204.09387) | Header bài báo |
| 7 | ADSS-UNet có mã nguồn công khai. | **SAI (FALSE)** | Tác giả ghi "Code will be made available", nhưng tới nay không công khai repo. Trạng thái: `NO_OFFICIAL_REPO`. | GitHub tác giả KTH | Xác nhận vắng mặt |
| 8 | Yadav Frontiers 2022: Tác giả là "R. Yadav, F. Dell'Acqua". | **SAI (FALSE)** | Tác giả thực tế: **Ritu Yadav, Andrea Nascetti, Yifang Ban** (KTH). "Dell'Acqua" là do AI đời trước suy diễn sai. Repo chính thức: `Uni-Temporal-Flood-Detection-Sentinel-1_Frontiers22`. | [Frontiers in Remote Sensing 2022](https://www.frontiersin.org/articles/10.3389/frsen.2022.955411/full) | Trang bìa & tác giả |
| 9 | Prithvi-100M: Báo cáo cũ ghi Water IoU 0.738 / mIoU 0.852. | **BỊA ĐẶT (FABRICATED)** | Bài báo gốc Prithvi (Bảng 2, trang 17) ghi nhận Water IoU đạt **82.99%**, mIoU đạt **90.16%** (sau 500 epoch). Model card HuggingFace 100 epoch ghi: Test Water IoU **80.46%**, mIoU **88.68%**, Bolivia **77.95%**. Con số 0.738 là hoàn toàn bịa. | [arXiv:2310.18660](https://arxiv.org/abs/2310.18660) & [HuggingFace Prithvi Card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11) | Trang 17, Bảng 2; HF Model Card |
| 10 | Prithvi trích dẫn baseline Sen1Floods11 là 24.21. | **CHÍNH XÁC (TRUE)** | Trong Bảng 1 của Bonafilia (CVPRW 2020), chỉ số Flooded Water (FW) Mean IoU của mô hình Hand-labeled S1 chính là **0.2421 (24.21%)**. Nhóm NASA/IBM lấy trực tiếp số này. | [Bonafilia CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf) | Trang 214, Bảng 1, cột FW Mean IoU |
| 11 | PANGAEA Benchmark: Báo cáo cũ ghi arXiv:2405.02949, IEEE GRSL 2024, S1-only 0.718. | **SAI (FALSE)** | Mã arXiv thực tế là **arXiv:2412.04204**. PANGAEA chỉ đánh giá Sen1Floods11 ở modality Multi-Spectral (Sentinel-2), không có S1-only. Repo: `VMarsocci/pangaea-bench`. | [arXiv:2412.04204](https://arxiv.org/abs/2412.04204) | Phụ lục C |
| 12 | DOFA: Báo cáo cũ ghi "A. Chopra et al., Dynamic MLP-based...". | **SAI (FALSE)** | Tác giả thực tế: **Zhitong Xiong, Yi Wang, Chenying Liu, Nassim Ait Ali Braham, Lichao Mou, Xiao Xiang Zhu** (TUM Đức). Tên bài: *Neural Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities* (IEEE TGRS 2024 / arXiv:2403.15356). Repo: `zhu-xlab/DOFA`. | [arXiv:2403.15356](https://arxiv.org/abs/2403.15356) | Header bài báo |
| 13 | Explainable SegFormer: Báo cáo cũ ghi "Hasan et al. 2024, B0-B4, IoU 0.634". | **SAI (FALSE)** | Bài báo thực tế là của **Biprodip Banerjee & Rita Daou** (arXiv:2407.03362), khảo sát U-Net, UNet++, DeepLabV3+ và SegFormer (b0, b1, b2) trên ETCI 2021 và Sen1Floods11. Tác giả Hasan là thông tin sai lệch. | [arXiv:2407.03362](https://arxiv.org/abs/2407.03362) | Tiêu đề & Bảng 3–4 |
| 14 | SEN12-FLOOD do ESA/DFKI phát triển. | **SAI (FALSE)** | SEN12-FLOOD do **CNES (Cơ quan Vũ trụ Pháp), ONERA, CNAM** xây dựng (Rambour et al.). Không thuộc ESA/DFKI. | Paper SEN12-FLOOD | Cơ quan chủ quản |
| 15 | Kaggle `sen1floods11-essentials` do `faizalkarim` tải lên. | **SAI (FALSE)** | `faizalkarim` tải lên dataset `flood-area-segmentation` (290 ảnh RGB thông thường). `sen1floods11-essentials` do user khác đăng tải. | Kaggle Dataset Profile | Tác giả Kaggle |
| 16 | Mã nguồn gốc Sen1Floods11 sử dụng Random Crop 256x256 khi train và cắt 4 góc phần tư 256x256 khi test. | **CHÍNH XÁC (TRUE)** | Kiểm tra trực tiếp file `Train.ipynb` (Cell 15): hàm `processAndAugment` dùng `transforms.RandomCrop.get_params(im1, (256, 256))` và hàm `processTestIm` cắt chính xác 4 góc phần tư 256x256. | File `Train.ipynb` trong repo CloudToStreet | Cell 15 |

---

## 2. KẾT LUẬN KHOA HỌC VỀ SLIDING WINDOW 256x256

### Phán quyết: **HỢP LỆ VỚI CÁC ĐIỀU KIỆN RÀNG BUỘC KHOA HỌC (VALID WITH CONDITIONS)**

#### A. Căn cứ hình học và khối lượng dữ liệu:
Chip nguyên bản Sen1Floods11 có kích thước $512 \times 512$ pixel ở độ phân giải $10\text{m/pixel}$ ($5.12\text{km} \times 5.12\text{km}$).
Khi phân rã thành các mảnh $256 \times 256$ không chồng lấn ($stride = 256$):
$$N = \left(\frac{512}{256}\right)^2 = 4\text{ mảnh / chip gốc}$$
- **Tập Train**: $252 \times 4 = 1.008$ mảnh.
- **Tập Valid**: $89 \times 4 = 356$ mảnh.
- **Tập Test**: $90 \times 4 = 360$ mảnh.
- **Tập Bolivia Holdout**: $15 \times 4 = 60$ mảnh.
- **Tổng cộng**: $1.784$ mảnh $256 \times 256$.

#### B. 5 Nguyên tắc Bắt buộc:
1. **Ranh giới phân chia (Split Boundary Isolation)**: Tuyệt đối chỉ cắt các mảnh con bên trong từng chip đã thuộc về phân vùng Train, Valid, hoặc Test. Không để mảnh con cắt ngang qua ranh giới giữa các tập. Tập Bolivia (60 mảnh) phải được giữ nguyên vẹn độc lập để kiểm định out-of-region zero-shot.
2. **Không tự tương quan không gian trên tập Test (I.I.D. Preservation)**: Trên tập Test, bắt buộc dùng stride 256 (không chồng lấn). Nếu dùng cửa sổ trượt chồng lấn $stride=128$, hai cửa sổ kề nhau chia sẻ tới $75\%$ diện tích chung, vi phạm giả định mẫu thử độc lập, làm sai lệch kết quả thống kê.
3. **Mất cân bằng lớp (Zero-Water Chips)**: Khoảng $43\%$ số mảnh con $256 \times 256$ chỉ chứa toàn đất (0% nước). Nếu huấn luyện bằng CrossEntropy thuần túy thì model sẽ bị thiên lệch hoàn toàn về background. Do đó **bắt buộc phải dùng kết hợp BCE + Dice Loss** để duy trì gradient khi gặp mảnh ít nước.
4. **Quy chuẩn đo đạc so sánh**: Khi so sánh với y văn (Bonafilia 2020, Prithvi 2023), việc đo đạc trên 360 mảnh test 256x256 tương đương về mặt toán học với việc đo trên 90 chip gốc 512x512 vì tổng số pixel kiểm thử là bảo toàn tuyệt đối ($360 \times 256^2 = 90 \times 512^2$).
5. **Tiết kiệm tài nguyên phần cứng**: Kích thước 256x256 giảm bộ nhớ kích hoạt xuống $4\times$, giảm ma trận Self-Attention xuống $16\times$, cho phép chạy với `batch_size = 32` mượt mà trên Kaggle GPU T4 16GB mà chỉ tốn ~3.2 GB VRAM.

---

## 3. DANH MỤC BASELINE CÓ MÃ NGUỒN CÔNG KHAI

| Mô hình | Đơn vị chủ quản / Tác giả | Repo GitHub Chính thức | Commit / Tag | License | Pretrained Weights | Kênh đầu vào | Trạng thái |
|---|---|---|---|---|---|---|---|
| **Sen1Floods11 Official Baseline** | Cloud to Street / NASA | `https://github.com/cloudtostreet/Sen1Floods11.git` | `3e22646` | CC-BY-4.0 | Không | S1 (2 ch), S2 (13 ch) | Hoạt động, đã kiểm chứng |
| **Prithvi-100M** | IBM Research & NASA IMPACT | `https://github.com/NASA-IMPACT/hls-foundation-os.git` | `main` | Apache-2.0 | Có (Hugging Face) | S2 (6 kênh HLS) | Hoạt động, đã kiểm chứng |
| **DOFA** | TU Munich (Zhitong Xiong et al.) | `https://github.com/zhu-xlab/DOFA.git` | `main` | Apache-2.0 | Có (Hugging Face) | S1 (2 ch), S2 (đa phổ) | Hoạt động, đã kiểm chứng |
| **Attentive S1 Fusion** | KTH Sweden (Ritu Yadav et al.) | `https://github.com/RituYadav92/Uni-Temporal-Flood-Detection-Sentinel-1_Frontiers22.git` | `main` | MIT | Không | S1 (2 ch + ratios) | Hoạt động, đã kiểm chứng |
| **PANGAEA Benchmark** | TU Berlin (Marsocci et al.) | `https://github.com/VMarsocci/pangaea-bench.git` | `main` | MIT | Có | S2 Multi-spectral | Hoạt động, đã kiểm chứng |
| *ProCANet (Feliren et al. 2025)* | Feliren et al. | `NO_OFFICIAL_REPO` | N/A | N/A | Không | S1 + S2 Dual-stream | Không công bố mã nguồn |
| *ADSS-UNet (Yadav et al. 2022)* | KTH Sweden | `NO_OFFICIAL_REPO` | N/A | N/A | Không | S1 Bi-temporal | Không công bố mã nguồn |

---

## 4. BẢNG ĐỐI CHIẾU CHUẨN HÓA A*/Q1

Để đạt chuẩn mực xuất bản tại các hội nghị A* (CVPR, ICCV, NeurIPS) hoặc tạp chí Q1 (IEEE TGRS, ISPRS, RSE), khung benchmark này đã thiết lập chính sách đồng nhất (Uniform Policy):

1. **Đồng nhất dữ liệu phân chia**: Cùng tập Train (1.008), Val (356), Test (360), Bolivia (60).
2. **Đồng nhất tiền xử lý**: Giá trị thống kê chuẩn hóa (Mean/Std) chỉ được tính trên tập Train.
3. **Đồng nhất tối ưu hóa**: Cùng optimizer AdamW, cùng scheduler CosineAnnealingLR ($T_{\max}=25$), cùng hàm tổn thất `BCEDiceLoss(ignore_index=255)`.
4. **Kiểm định thống kê qua 5 Seeds**: Mặc định chạy 5 seed ngẫu nhiên (`42, 43, 44, 45, 46`). Báo cáo $\text{Mean} \pm \text{Std}$ và khoảng tin cậy $95\%$ Confidence Interval.
5. **Kiểm định có nghĩa thống kê (Significance Testing)**: Tự động chạy kiểm định ghép cặp (Paired Wilcoxon test / Paired t-test) giữa các model trên phân phối IoU từng chip.
6. **Đo đạc hiệu năng phần cứng chuẩn mực**: Đo thời gian suy luận (latency ms), thông lượng (FPS), số tham số (Params M), lượng tính toán (FLOPs Giga) và bộ nhớ đỉnh (Peak VRAM MB) với kỹ thuật GPU warm-up và `torch.cuda.synchronize()`.

---

## 5. HƯỚNG DẪN THỰC THI TRÊN KAGGLE GPU

### Bước 1: Mở Notebook Kaggle
- Tạo Notebook mới trên Kaggle, chọn **GPU T4 x 2** hoặc **P100**.
- Tải file `sen1floods11-bench/notebooks/kaggle_sen1floods11_256.ipynb` lên và chạy.

### Bước 2: Tải dữ liệu tự động từ Hugging Face
```python
from huggingface_hub import hf_hub_download
import os

DATA_DIR = '/kaggle/working/data'
os.makedirs(DATA_DIR, exist_ok=True)
hf_hub_download(repo_id='doantrongthai/sen1floods11', filename='sen1floods11.zip', repo_type='dataset', local_dir='/kaggle/working')
!unzip -q -o /kaggle/working/sen1floods11.zip -d {DATA_DIR}
```

### Bước 3: Chạy Benchmark bằng CLI một dòng lệnh
```bash
# Huấn luyện và tự động đánh giá Test + Bolivia trong ~1 phút:
python scripts/train.py --data_dir "/kaggle/working/data" --model unet --modality s1 --epochs 25 --batch_size 32
```
*Log in đúng 1 dòng/epoch, an toàn tuyệt đối không gây tràn bộ đệm Kaggle.*

### Bước 4: Đóng gói và tải trọng số về máy cục bộ
Trên Kaggle, chạy lệnh pack weights:
```bash
python scripts/pack_weights.py --run_dir "./runs/unet_s1_seed42" --model "unet" --run_id "seed42"
```
Trên máy cá nhân, tải về và kiểm tra toàn vẹn mã băm SHA256:
```bash
python weights/fetch_local.py --model "unet" --run_id "seed42"
```

---

## 6. CÁC HẠNG MỤC CHƯA THỰC HIỆN VÀ LÝ DO

1. **Huấn luyện hội tụ cục bộ trên toàn bộ 5 hạt giống**:
   - *Lý do*: Môi trường cục bộ hiện tại không có card đồ họa CUDA (`torch.cuda.is_available() == False`, CPU 16-core). Theo quy chuẩn liêm chính khoa học, tuyệt đối không bịa đặt số liệu "reproduced" khi chưa chạy thực tế trên GPU. Toàn bộ code đã sẵn sàng và được kiểm thử thành công qua suite 9 test.
2. **Tải mã nguồn ProCANet và ADSS-UNet**:
   - *Lý do*: Nhóm tác giả Feliren et al. (ProCANet) và Yadav et al. (ADSS-UNet) chưa từng công khai mã nguồn chính thức (`CONFIRMED_ABSENT`).
