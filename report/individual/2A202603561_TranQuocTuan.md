# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**  2A202603561
- **Họ và tên:** Trần Quốc Tuấn
- **SEED** (in ở ô đầu notebook): Lab Ngày 1 — Một bài toán, bốn thế hệ AI: **Lọc tin nhắn rác (Spam / Ham)

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

<img width="889" height="489" alt="Unknown" src="https://github.com/user-attachments/assets/5c7fd509-2067-4711-8e6c-8bc755876f3a" />

## 3. Dự đoán và thực tế

1. Dự đoán về nhóm khó nhất trong bộ dữ liệu Stress Test (hard_eval)

Dự đoán: Bạn cho rằng nhóm manipulation (Thao túng / lừa đảo) sẽ là nhóm khó nhất đối với mô hình.

Thực tế: Nhóm tệ nhất của mô hình BoW thực chất là natural_spam (Spam giống tự nhiên) với tỉ lệ chính xác là 0.0% (trong khi nhóm manipulation đạt 12.5%).

Nguyên nhân lệch: Nhóm Spam giống tự nhiên được cố ý viết bằng ngôn từ rất đời thường, đóng vai bạn bè quen thuộc và hoàn toàn né tránh các từ khóa mang tính quảng cáo mạnh (như FREE, WIN, Claim). Vì Bag-of-Words (BoW) của mô hình 1 chỉ học dựa trên tần suất xuất hiện của từ khóa đơn lẻ mà không hiểu ngữ cảnh toàn câu, nó hoàn toàn bị đánh lừa và đoán sai 100% các tin nhắn thuộc nhóm này thành ham.

2. Dự đoán về các mô hình nằm trên đường tối ưu Pareto (tradeoff)

Dự đoán: Bạn dự đoán các mô hình Embedding và LLM sẽ nằm trên đường Pareto, còn các phương pháp LLM sẽ nằm bên phải vạch giới hạn SLA 50ms.

Thực tế: Đường Pareto thực tế bao gồm các mô hình: BoW + LogReg (tự code), BoW + MLP (deep learning), và Encoder (MiniLM) + LogReg. Trong khi đó, các mô hình vượt (chậm hơn) vạch SLA 50ms thực tế gồm Encoder + kNN (p95 = 77.15 ms) và toàn bộ các phiên bản LLM (với độ trễ p50/p95 từ vài trăm đến vài nghìn ms).

Nguyên nhân lệch: Do tập dữ liệu thử nghiệm SUB khá nhỏ (40 tin), các mô hình BoW đơn giản nhưng cực kỳ nhanh (độ trễ dưới 1ms) đã thể hiện hiệu năng F1 rất cạnh tranh và chiếm ưu thế tuyệt đối về mặt tốc độ, đẩy các mô hình LLM phức tạp và chậm chạp ra khỏi đường tối ưu Pareto.

3. Dự đoán về điểm F1-score của mô hình cơ bản (ml_f1)

Dự đoán: Bạn dự đoán F1-score sẽ đạt khoảng 0.85 – 0.93 và Precision sẽ cao hơn Recall do sự mất cân bằng dữ liệu.
- Thực tế: F1-score trên tập Test đạt 0.882, với chỉ số **Precision đạt tuyệt đối 1.000 (không chặn nhầm tin Ham nào) và Recall đạt 0.788.
- Nguyên nhân lệch: Mặc dù khoảng dự đoán F1 của bạn rất chính xác, nhưng mức lệch nằm ở việc Precision đạt tới mức tuyệt đối 1.000 trên tập Test thực tế. Điều này cho thấy mô hình Logistic Regression tự code của bạn đã thiết lập một ranh giới phân loại rất an toàn (chỉ những tin nhắn có đặc trưng spam cực kỳ rõ ràng mới bị gán nhãn 1), dẫn đến việc bỏ sót khá nhiều tin nhắn spam thực tế (Recall thấp hơn nhiều) nhưng đảm bảo độ chính xác tuyệt đối khi đã ra quyết định chặn.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
