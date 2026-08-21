# Bao Cao Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Ho va ten | Le Tri Tung |
| MSSV | 2A202601458 |
| Lop / Khoa | K4 |
| Repo GitHub | https://github.com/ltt230205/TRACK2_Day21_2A202601458_LeTriTung |
| Ngay nop | 21/08/2026 |

---

## 1. Bo Sieu Tham So Da Chon va Ly Do

| Lan chay | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 100 | 0.10 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.10 | 5 | 0.7149 | 0.8740 |
| 4 | 200 | 0.05 | 3 | 0.7014 | 0.8740 |

**Bo sieu tham so da chon:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Ly do:** Cau hinh nay co `f1_score` cao nhat tren tap holdout va vuot nguong quality gate `0.65`. Lan 1 co accuracy cao hon mot chut, nhung F1 thap hon, nen toi khong chon theo accuracy. Ket qua nay cho thay voi du lieu mat can bang, accuracy co the on dinh trong khi kha nang bat dung lop thu nhap cao thay doi ro hon qua F1.

---

## 2. Vi Sao Nguong Chat Luong Dat Tren F1 Chu Khong Phai Accuracy

Bo du lieu Adult bi mat can bang: chi khoang 24.8% mau thuoc lop thu nhap tren 50K. Neu mot mo hinh luon du doan "thu nhap thap", accuracy van co the dat khoang 0.752, nhung mo hinh do khong phat hien duoc truong hop thu nhap cao nao. Vi vay accuracy de gay hieu nham trong bai toan nay. F1 cua lop duong ket hop precision va recall, phan anh tot hon kha nang tim dung nhom thu nhap cao. Khi tinh F1, toi dung mac dinh `f1_score(y_eval, preds)` cho lop duong, khong dung `weighted` vi lop da so co the keo diem len.

---

## 3. Kho Khan Gap Phai va Cach Giai Quyet

| Kho khan | Nguyen nhan | Cach giai quyet |
|---|---|---|
| MLflow loi `pkg_resources` tren Python 3.12 | `setuptools` moi khong con cung cap module nay theo cach MLflow 2.13 can | Pin `setuptools<81` trong `requirements.txt` |
| Workflow can cloud credentials | CI/CD va DVC remote phu thuoc bucket, service account, VM va GitHub Secrets | Viet workflow doc secrets dung chuan GCP, con gia tri that se cau hinh tren GitHub |
| Du lieu mat can bang lam accuracy kho dien giai | Lop thu nhap thap chiem da so | Chon F1 cua lop duong lam chi so chinh va quality gate |

---

## 4. So Sanh Buoc 2 va Buoc 3

| | f1_score | accuracy |
|---|---:|---:|
| Buoc 2 (chi `train_batch1`) | 0.7149 | 0.8740 |
| Buoc 3 (them `train_batch2`) | 0.7354 | 0.8820 |

**Nhan xet:** Sau khi them `train_batch2`, F1 tang tu 0.7149 len 0.7354 va accuracy tang tu 0.8740 len 0.8820. Batch moi co cung phan phoi voi batch dau, nen muc tang khong qua lon, nhung pipeline van chung minh duoc quy trinh tu du lieu moi den huan luyen lai va tao model moi.
