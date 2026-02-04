# DUPGT-CDR: Deep User Preference Gating Transfer for Cross-domain Recommendation
DUPGT-CDR(Deep User Preference Gating Transfer for Cross-domain Recommendation)の実装リポジトリです。

## Results

本手法は、クロスドメイン推薦における従来手法と比較して、以下の性能向上を達成しています。

### 比較対象手法
- **CMF**: [Relational Learning via Collective Matrix Factorization (KDD 2008)](https://dl.acm.org/doi/pdf/10.1145/1401890.1401969?casa_token=S9kvmlp1bxEAAAAA:v96uHthvspO1ahgCZ1htH8sGl2voMvREqwXVYGf3X4WbvYXaD7tX1OsfXhx4k126HSOOtsbcbf9q)
- **EMCDR**: [Cross-Domain Recommendation: An Embedding and Mapping Approach (IJCAI 2017)](https://www.ijcai.org/Proceedings/2017/0343.pdf)
- **PTUPCDR**: [Personalized Transfer of User Preferences for Cross-domain Recommendation (WSDM 2022)](https://dl.acm.org/doi/pdf/10.1145/3488560.3498392?casa_token=fMj33BdRcdoAAAAA:7iA-ORhh02jV0wY2bPg3keZVcDxAXt5q8hM-9JM8oKrTFj7caBd-HUOICs6gfrIV6tch8NpcYYOC)
- **DPTUPCDR**: [Tackling cold-start with deep personalized transfer of user preferences for cross-domain recommendation (Int J Data Sci Anal 20, 121–130 (2025))](https://doi.org/10.1007/s41060-023-00467-9)
- **MIMNet**: [Multi-interest Meta Network with Multi-granularity Target-guided Attention for cross-domain recommendation (Int J Neurocomputing vol. 620, p. 129208 (2025))](https://doi.org/10.1016/j.neucom.2024.129208)

## Features

DUPGT-CDRは、クロスドメイン推薦における以下の特徴を持っています。

### 主要機能
- **二重ブリッジング機構**: ユーザーの高頻度嗜好と低頻度嗜好を別々にモデル化し、より精緻な嗜好転移を実現
- **ゲーティングMLP**: ユーザー埋め込み、アイテム埋め込み、および二つのブリッジング表現を統合し、適応的な嗜好転移を実現
- **深層メタネットワーク**: ユーザーの履歴情報から効果的に嗜好パターンを抽出
- **段階的学習プロセス**: ソースドメイン学習、ターゲットドメイン学習、マッピング学習、メタ学習の4段階で構成

### データセット
Amazon-5coresデータセットの以下のカテゴリを使用:
- Movies & TV Shows
- CDs & Records  
- Books

詳細は[Amazon Reviews Dataset](http://jmcauley.ucsd.edu/data/amazon/links.html)を参照してください。

### タスク構成
3つのクロスドメインタスクをサポート:
1. Movies & TV → CDs & Vinyl
2. Books → Movies & TV
3. Books → CDs & Vinyl

## Requirement

以下の環境およびパッケージが必要です。

- Python 3.6以上
- PyTorch 1.0以上
- TensorFlow
- Pandas
- NumPy
- Tqdm

## Installation

1. リポジトリのクローン
```bash
git clone https://github.com/yourusername/DUPGT-CDR.git
cd DUPGT-CDR
```

2. 必要なパッケージのインストール
```bash
pip install torch tensorflow pandas numpy tqdm
```

3. データセットの準備
Amazon-5coresデータセットをダウンロードし、`../data/`ディレクトリに配置してください。

## Usage

### データの前処理

#### 中間データの生成
```bash
python entry.py --process_data_mid 1
```

#### 学習用データの生成
```bash
python entry.py --process_data_ready 1
```

### モデルの学習と評価

基本的な実行方法:
```bash
python entry.py --task 1 --ratio [0.8, 0.2] --epoch 15 --lr 0.01 --gpu 0
```

#### パラメータ説明
- `--task`: タスク番号(1, 2, 3のいずれか)
  - 1: Movies & TV → CDs & Vinyl
  - 2: Books → Movies & TV
  - 3: Books → CDs & Vinyl
- `--ratio`: ソースドメインとターゲットドメインのデータ分割比率
  - デフォルト: [0.8, 0.2]
  - その他の設定: [0.5, 0.5], [0.2, 0.8]
- `--epoch`: 学習エポック数(デフォルト: 15)
- `--lr`: 学習率(デフォルト: 0.01)
- `--gpu`: 使用するGPU番号(デフォルト: 0)
- `--seed`: 乱数シード(デフォルト: 2020)

### 実行例

タスク1を異なるデータ分割比率で実行:
```bash
# ソース80%、ターゲット20%
python entry.py --task 1 --ratio [0.8, 0.2]

# ソース50%、ターゲット50%
python entry.py --task 1 --ratio [0.5, 0.5]

# ソース20%、ターゲット80%
python entry.py --task 1 --ratio [0.2, 0.8]
```

## Note

### モデル構成
- 埋め込み次元: 10
- メタネットワーク次元: 50
- フィールド数: 2
- 重み減衰: 0

### 学習プロセス
モデルは以下の4段階で学習を行います:
1. ソースドメインでの基礎学習
2. ターゲットドメインでの基礎学習  
3. ドメイン間のマッピング学習
4. メタ学習によるクロスドメイン推薦

### 設定ファイル
`config.json`でタスクごとのハイパーパラメータを設定可能です。各タスクに対して、ユーザー数、アイテム数、バッチサイズなどが個別に設定されています。

## Author
- 清水 雄介
- 所属: 同志社大学大学院 理工学研究科 情報工学専攻 知的システムデザイン研究室
- Email: shimizu.yusuke@mikilab.doshisha.ac.jp