資料清洗：
針對賣方＆品項
全形轉半形
全部轉小寫
去除前後空白、重複空白只留一個
保留英文、中文、空白、-/()+&%_
針對 store name 移除「股份有限公司/分公司」等字眼
對空白的 store or item name 給予 [UNK_STORE], [UNK_ITEM] 的標籤

針對每列發票
刪掉金額 <= 0 的列（這些通常是折扣）

LLM 預標註：
Model: Claude Sonnet 4.6
Prompt: 深度學習實作與應用
* 有再人工核對過

資料分布：
分類標籤改為六大類：[飲食, 交通, 購物, 娛樂, 醫療健康, 教育]
（因為「其他」和「3C電子」佔太少啦 -> 3C電子併入購物）

label
label_id
飲食
0
交通
1
購物
2
娛樂
3
教育
4
醫療健康
5



根據消費金額評估消費金額區間：

金額區間
price_bucket
< 50
VERY_LOW
50 ~ 150
LOW
150 ~ 500
MID
500 ~ 1500
HIGH
> 1500
VERY_HIGH



分類結果：
https://drive.google.com/file/d/1jcws715rEOgb8JYoI1kdjwza9SPbXHaR/view?usp=sharing


Distribution:
總共有 2613 筆發票明細（3 人）

標籤
筆數
佔比
飲食
2153
0.823957
交通
164
0.062763
購物
131
0.050133
娛樂
72
0.027555
教育
69
0.026406
醫療健康
24
0.009185



分類結果：
BERT -> [CLS] 品名 [SEP] 店名 金額區間 [SEP]
Embedding: https://drive.google.com/file/d/14os0npHWrMKusa008qsAzD4Q3-rH3QPZ/view?usp=sharing



precision
recall
f1-score
support
飲食
0.98
0.99
0.99
386
交通
1.00
1.00
1.00
9
購物
0.84
0.87
0.86
31
娛樂
1.00
0.75
0.86
8
教育
1.00
0.88
0.93
16
醫療健康
1.00
0.80
0.89
5










accuracy




0.98
455
macro avg
0.97
0.88
0.92
455
weighted avg
0.98
0.98
0.98
455



Benchmark
TF-IDF + Logistic Regression



precision
recall
f1-score
support
飲食
0.94
0.72
0.81
386
交通
1.00
0.33
0.50
9
購物
0.31
0.61
0.41
31
娛樂
0.43
0.38
0.40
8
教育
0.12
0.38
0.18
16
醫療健康
0.05
0.40
0.09
5










accuracy




0.68
455
macro avg
0.47
0.47
0.40
455
weighted avg
0.85
0.68
0.74
455


TF-IDF + MLP



precision
recall
f1-score
support
飲食
0.90
0.89
0.89
386
交通
1.00
0.33
0.50
9
購物
0.33
0.48
0.39
31
娛樂
1.00
0.38
0.55
8
教育
0.26
0.38
0.31
16
醫療健康
0.00
0.00
0.00
5










accuracy




0.81
455
macro avg
0.58
0.41
0.44
455
weighted avg
0.83
0.81
0.81
455


BERT -> [CLS] 品名 [SEP] 店名 金額區間 [SEP]



precision
recall
f1-score
support
飲食
0.96
1.00
0.98
386
交通
0.78
0.78
0.78
9
購物
0.86
0.61
0.72
31
娛樂
1.00
0.75
0.86
8
教育
1.00
0.75
0.86
16
醫療健康
0.67
0.80
0.73
5










accuracy




0.95
455
macro avg
0.88
0.78
0.82
455
weighted avg
0.95
0.95
0.95
455



