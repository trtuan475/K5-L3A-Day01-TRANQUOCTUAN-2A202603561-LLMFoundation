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

- Dự đoán: Bạn cho rằng nhóm manipulation (Thao túng / lừa đảo) sẽ là nhóm khó nhất đối với mô hình.

- Thực tế: Nhóm tệ nhất của mô hình BoW thực chất là natural_spam (Spam giống tự nhiên) với tỉ lệ chính xác là 0.0% (trong khi nhóm manipulation đạt 12.5%).

- Nguyên nhân lệch: Nhóm Spam giống tự nhiên được cố ý viết bằng ngôn từ rất đời thường, đóng vai bạn bè quen thuộc và hoàn toàn né tránh các từ khóa mang tính quảng cáo mạnh (như FREE, WIN, Claim). Vì Bag-of-Words (BoW) của mô hình 1 chỉ học dựa trên tần suất xuất hiện của từ khóa đơn lẻ mà không hiểu ngữ cảnh toàn câu, nó hoàn toàn bị đánh lừa và đoán sai 100% các tin nhắn thuộc nhóm này thành ham.

2. Dự đoán về các mô hình nằm trên đường tối ưu Pareto (tradeoff)

- Dự đoán: Bạn dự đoán các mô hình Embedding và LLM sẽ nằm trên đường Pareto, còn các phương pháp LLM sẽ nằm bên phải vạch giới hạn SLA 50ms.

- Thực tế: Đường Pareto thực tế bao gồm các mô hình: BoW + LogReg (tự code), BoW + MLP (deep learning), và Encoder (MiniLM) + LogReg. Trong khi đó, các mô hình vượt (chậm hơn) vạch SLA 50ms thực tế gồm Encoder + kNN (p95 = 77.15 ms) và toàn bộ các phiên bản LLM (với độ trễ p50/p95 từ vài trăm đến vài nghìn ms).

- Nguyên nhân lệch: Do tập dữ liệu thử nghiệm SUB khá nhỏ (40 tin), các mô hình BoW đơn giản nhưng cực kỳ nhanh (độ trễ dưới 1ms) đã thể hiện hiệu năng F1 rất cạnh tranh và chiếm ưu thế tuyệt đối về mặt tốc độ, đẩy các mô hình LLM phức tạp và chậm chạp ra khỏi đường tối ưu Pareto.

3. Dự đoán về điểm F1-score của mô hình cơ bản (ml_f1)

- Dự đoán: Bạn dự đoán F1-score sẽ đạt khoảng 0.85 – 0.93 và Precision sẽ cao hơn Recall do sự mất cân bằng dữ liệu.
- Thực tế: F1-score trên tập Test đạt 0.882, với chỉ số **Precision đạt tuyệt đối 1.000 (không chặn nhầm tin Ham nào) và Recall đạt 0.788.
- Nguyên nhân lệch: Mặc dù khoảng dự đoán F1 của bạn rất chính xác, nhưng mức lệch nằm ở việc Precision đạt tới mức tuyệt đối 1.000 trên tập Test thực tế. Điều này cho thấy mô hình Logistic Regression tự code của bạn đã thiết lập một ranh giới phân loại rất an toàn (chỉ những tin nhắn có đặc trưng spam cực kỳ rõ ràng mới bị gán nhãn 1), dẫn đến việc bỏ sót khá nhiều tin nhắn spam thực tế (Recall thấp hơn nhiều) nhưng đảm bảo độ chính xác tuyệt đối khi đã ra quyết định chặn.

## 4. Phân tích 50 case khó

## 1. Mô hình bị tụt giảm hiệu năng nhiều nhất

- **Mô hình bị tụt nhiều nhất:** Mô hình **1. BoW + LogReg (tự code)** và **2. BoW + MLP (deep learning)**.
  - **BoW + LogReg:** Điểm F1 trên tập kiểm thử thường (SUB) đạt 0.889, nhưng khi đối mặt với 50 case khó, F1 giảm thê thảm xuống chỉ còn 0.294 (giảm gần 0.60 điểm F1).
  - **BoW + MLP:** F1 từ 0.947 rơi xuống còn 0.343 (giảm 0.60 điểm F1).

- **Nhóm case gây khó khăn lớn nhất:** Nhóm **Spam giống tự nhiên (natural_spam)** với tỉ lệ đoán đúng của mô hình BoW là 0.0% và nhóm **Né từ khóa (obfuscation)** với tỉ lệ đoán đúng chỉ 17%.

- **Nguyên nhân tập test thường quá lạc quan:** Tập test thông thường được chia ngẫu nhiên từ cùng một phân phối dữ liệu với tập train. Do đó, các từ khóa đặc trưng của spam (như FREE, call, txt, claim) xuất hiện lặp đi lặp lại một cách rõ ràng trong cả hai tập, giúp các mô hình đơn giản dễ dàng ghi nhớ và đạt điểm số cao vượt trội.

## 2. Hai trường hợp sai cụ thể và cơ chế gây lỗi

### Trường hợp 1: Tin nhắn thuộc nhóm "Spam giống tự nhiên" (Thật = spam | BoW đoán = ham)

- **Nội dung tin nhắn:** "Hey it's Jess from the gym 😊 we're doing a members-only thing this week, 3 months for the price of 1, just reply JOIN and I'll sort it"

- **Cơ chế khiến model đoán sai:**
  - Tin nhắn này được viết rất tự nhiên, đóng vai một người quen thân thiết tại phòng tập thể hình (gym) và không chứa bất kỳ từ khóa đẩy spam mạnh truyền thống nào có trong từ điển học được (như win, cash, award, urgent, reply YES).
  - Đồng thời, nó chứa nhiều từ khóa kéo về nhãn ham mạnh (như i, me, my, we).
  - Do mô hình Bag-of-Words (BoW) chỉ đếm tần suất xuất hiện của các từ riêng lẻ một cách độc lập mà không hiểu được cấu trúc ngữ nghĩa quảng cáo/lôi kéo tiềm ẩn, mô hình hoàn toàn tin rằng đây là tin nhắn cá nhân thông thường và gán nhãn ham.

### Trường hợp 2: Tin nhắn thuộc nhóm "Tưởng spam nhưng không phải" (Thật = ham | BoW đoán = spam)

- **Nội dung tin nhắn:** "Congrats!!! You WON the bet lol, I owe you a beer. Call me later"

- **Cơ chế khiến model đoán sai:**
  - Tin nhắn này chứa một loạt các từ khóa mang trọng số đẩy về phía spam cực mạnh đã học được từ tập train: congrats, WON và Call.
  - Tuy nhiên, ngữ cảnh thực tế ở đây chỉ là hai người bạn đang trò chuyện và cá cược vui vẻ với nhau ("lol, I owe you a beer").
  - Do mô hình BoW bị giới hạn trong việc liên kết các token độc lập và không thể phân tích được ngữ cảnh thân mật của câu, sự hiện diện của các từ viết hoa mạnh lập tức kích hoạt bộ lọc và khiến mô hình chặn nhầm một tin nhắn hoàn toàn bình thường.
## 5. Thử thách cá nhân

1.  Kết quả
- Độ chính xác: Hai phương pháp có thể đạt F1 tương đương vì cùng sử dụng một mô hình và cùng nhiệm vụ phân loại. Việc bổ sung trường reason không đảm bảo cải thiện khả năng phân loại.
- Số lượng token và chi phí: JSON thường tiêu tốn nhiều output token hơn vì phải tạo thêm cấu trúc JSON và phần giải thích. Điều này có thể làm tăng chi phí và thời gian phản hồi.
- Khả năng giải thích: JSON có ưu thế vì cung cấp lý do phân loại, giúp người dùng kiểm tra và hiểu kết quả. Tuy nhiên, cần xử lý trường hợp JSON không hợp lệ để tránh ảnh hưởng đến hệ thống.
2. Kết luận
Nếu mục tiêu chỉ là phát hiện SMS spam, đầu ra một từ là lựa chọn phù hợp hơn vì ngắn gọn, tiết kiệm token và dễ xử lý. Ngược lại, JSON phù hợp với những hệ thống cần khả năng giải thích kết quả, hỗ trợ kiểm duyệt hoặc cung cấp thông tin cho người dùng.

## 6. Phương án cho sản phẩm thật

1. Phân tích các mô hình đạt SLA (p95 < 50 ms)

Dựa vào bảng kết quả thực nghiệm, dưới đây là các phương án đạt tiêu chuẩn thời gian phản hồi:

- 1. BoW + LogReg (tự code):
  - p95 latency:0.17 ms/tin (vô cùng nhanh, phù hợp cho hệ thống thời gian thực).
  - F1-Score (Test):** 0.882 (độ chính xác tương đối khá, tuy nhiên dễ bị đánh lừa bởi các tin nhắn phức tạp hoặc né từ khóa).
  - Tốc độ xử lý:** 4,070.18 tin/giây.

- 2. BoW + MLP (Deep Learning):
  - p95 latency:** 0.43 ms/tin.
  - F1-Score (Test): 0.933.
  - Tốc độ xử lý: 34,610.74 tin/giây (tốc độ tối ưu nhờ tận dụng tính toán song song và vector hóa).

- 3b. Encoder (MiniLM) + LogReg:
  - p95 latency: 41.64 ms/tin (nằm sát giới hạn dưới 50 ms).
  - F1-Score (Test): 0.952 (độ chính xác cao nhất trong nhóm đạt SLA).

- 3c. Encoder (MiniLM) + MLP head:**
  - p95 latency: 39.58 ms/tin.
  - F1-Score (Test): 0.943.

2. Lựa chọn phương án tối ưu cho sản phẩm thực tế

Để đạt cả độ trễ và chất lượng vượt trội, phương án 3b (Encoder MiniLM + LogReg) là lựa chọn tối ưu nhất:

- Lý do: Đạt F1 = 0.952 (tương đương với các mô hình LLM lớn và nhỏ không có prompt phòng thủ) nhưng có độ trễ p95 chỉ 41.64 ms, đáp ứng nghiêm ngặt cam kết dịch vụ (dưới 50 ms).
- Đánh đổi: Chúng ta phải đánh đổi bằng tài nguyên hạ tầng phần cứng. Mô hình này yêu cầu một máy chủ có GPU hoặc CPU đủ mạnh để chạy mô hình Transformer MiniLM (~90MB) liên tục nhằm sinh Embedding cho từng tin nhắn đầu vào trước khi phân loại.

 3. Ước tính chi phí một tháng nếu dùng LLM

Nếu sử dụng mô hình LLM phân loại trực tiếp thông qua API với giả định dùng mô hình nhỏ giá rẻ là gpt-6-luna (chi phí: $0.10/1M input tokens và $0.50/1M output tokens):

- Quy mô hệ thống: 10.000.000 SMS/ngày → 300.000.000 SMS/tháng.
- Cấu hình mỗi tin: 100 token input và 1 token output.
- Chi phí Input/ngày:

  10.000.000 × 100 = 1.000.000.000 tokens

  Tương đương 1.000 triệu tokens.

  Chi phí = 1.000 × $0.10 = $100/ngày.

- Chi phí Output/ngày:

  10.000.000 × 1 = 10.000.000 tokens

  Tương đương 10 triệu tokens.

  Chi phí = 10 × $0.50 = $5/ngày.

- Tổng chi phí một ngày:** $100 + $5 = $105/ngày.
- Tổng chi phí một tháng (30 ngày):

  $105 × 30 = $3,150/tháng.

## 7. AI log và điều đã học

Trước đây, tôi nghĩ mô hình có F1-score cao trên tập Test thông thường sẽ hoạt động tốt trong thực tế. Tuy nhiên, qua Stress Test, tôi nhận ra mô hình có thể giảm hiệu năng đáng kể khi gặp những tin nhắn spam được viết tự nhiên hoặc cố tình né từ khóa.

Tôi cũng hiểu rằng lựa chọn mô hình không chỉ dựa vào độ chính xác mà còn phải cân nhắc độ trễ, tốc độ xử lý và chi phí triển khai. Đặc biệt, việc kết hợp mô hình nhẹ với LLM thông qua Router có thể giúp cân bằng giữa chất lượng phân loại và hiệu quả vận hành thực tế
