> [!WARNING]
> **開発途上のソフトウェア**
> 
> このソフトウェアは現在開発中です。本格的な製造環境での使用前に十分なテストと検証を行ってください。
> 
> - プロトタイピング・評価用途
> - 製造環境での使用は慎重に検討
> - 機能は予告なく変更される可能性があります

> [!NOTE]
> **AI支援開発**
> 
> このプロジェクトは生成AI（Claude）を活用して開発されています。
> 
> - コード生成・レビュー
> - アルゴリズム設計支援
> - ドキュメント作成
> - トラブルシューティング

---

# 電子部品カウントシステム

## 概要

このシステムは、電子電気関連機器の製造工場において、チップ抵抗やトランジスタなどの小さな電子部品の数を画像認識技術により自動カウントするためのアプリケーションです。

## 特徴

- **高精度検出**: 複数のアルゴリズム（輪郭検出、テンプレートマッチング、YOLOv8）を併用
- **品質保証**: アルゴリズム間の一致度による品質スコア算出
- **画像補正機能**: リアルタイム画像補正により悪条件下でも高精度検出
- **直感的GUI**: 工場作業者が簡単に使用できるインターフェース
- **ローカル実行**: インターネット接続不要で動作
- **結果保存**: JSON形式での検出結果保存機能

## システム要件

- Python 3.8以上
- Windows / macOS / Linux対応
- メモリ: 4GB以上推奨
- CPU: マルチコア推奨（画像処理のため）

## インストール手順

### 1. 仮想環境の作成

```fish
# プロジェクトディレクトリに移動
cd counter

# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
# Windows の場合:
venv\Scripts\activate

# macOS/Linux (fishシェル) の場合:
source venv/bin/activate.fish
```

### 2. 依存関係のインストール

```fish
# 必要なパッケージをインストール
pip install -r requirements.txt

# YOLOv8モデルの初回ダウンロード（自動実行）
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. 動作確認

```fish
# アプリケーションを起動
python3 main.py
```

## 使用方法

### 基本的な使用手順

1. **アプリケーション起動**
   - `python3 main.py`または`./run_counter.fish`でGUIアプリケーションが起動します

2. **画像の選択**
   - 「画像を選択」ボタンをクリック
   - 部品が写った画像ファイルを選択（JPG、PNG、BMP、TIFF対応）

3. **画像補正（必要に応じて）**
   - 明度: 0.5-2.0の範囲で調整
   - コントラスト: 0.5-3.0の範囲で調整
   - ガンマ補正: 0.5-2.0の範囲で調整
   - ヒストグラム均一化: 照明ムラの改善
   - シャープネス強化: エッジの強調
   - ノイズ除去: ノイズの軽減
   - リセットボタンで元画像に戻す

4. **設定の調整**
   - 信頼度閾値: 検出の厳しさを調整（0.1-1.0）
   - 検出方法: 使用するアルゴリズムを選択

5. **部品カウント実行**
   - 「部品カウント実行」ボタンをクリック
   - 処理完了まで数秒～数十秒待機

6. **結果の確認**
   - 検出された部品数が表示されます
   - 画像には検出された部品がボックスで囲まれます
   - 詳細な結果が下部のテキストエリアに表示されます

7. **結果の保存**
   - 「結果を保存」ボタンでJSON形式で結果を保存可能

### 品質スコアについて

- **0.8以上**: 高品質（信頼できる結果）
- **0.5-0.8**: 中品質（確認推奨）
- **0.5未満**: 低品質（再撮影を推奨）

## 推奨撮影条件

### 照明条件
- 均一な照明（影を避ける）
- 十分な明るさ
- 反射やハレーションを避ける

**💡 ヒント**: 照明条件が悪い場合は、画像補正機能で改善できます

### 撮影角度
- 真上からの撮影を推奨
- 部品が重なっていない状態
- 背景とのコントラストが明確

### 画像品質
- 解像度: 1024x1024ピクセル以上推奨
- フォーカス: 明確なピント
- ノイズ: 低ノイズ

## トラブルシューティング

### よくある問題と解決法

#### 1. アプリケーションが起動しない
```fish
# 仮想環境が有効化されているか確認
which python3  # macOS/Linux (fishシェル)
where python  # Windows

# 依存関係を再インストール
pip install -r requirements.txt --force-reinstall
```

#### 2. tkinterエラーが発生する（macOS）
```fish
# HomeBrew環境でtkinterサポートを追加
brew install python-tk

# 仮想環境を再作成
rm -rf venv
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

#### 3. 検出精度が低い
- **画像補正を試す**:
  - 暗い画像: 明度を1.2-1.5に調整
  - コントラストが低い: コントラストを1.5-2.0に調整
  - 照明ムラ: ヒストグラム均一化を有効化
  - ぼやけた画像: シャープネス強化を有効化
  - ノイズが多い: ノイズ除去を有効化
- 撮影条件を改善（照明、角度、フォーカス）
- 信頼度閾値を調整
- 複数のアルゴリズムを併用（"all"選択）

#### 4. 処理が遅い
- 画像サイズを小さくする（システムが自動リサイズ）
- YOLOモデルを無効化（"contour"または"template"選択）

#### 5. メモリ不足エラー
- 画像サイズを小さくする
- 他のアプリケーションを終了
- config.pyの`MAX_IMAGE_SIZE`を調整

## 設定のカスタマイズ

`config.py`ファイルを編集することで、以下の設定を調整できます：

```python
# 検出設定
CONFIDENCE_THRESHOLD = 0.5  # 信頼度閾値
MIN_COMPONENT_SIZE = 10     # 最小部品サイズ

# 処理設定
MAX_IMAGE_SIZE = (1024, 1024)  # 最大画像サイズ
YOLO_DEVICE = "cpu"            # 使用デバイス
```

## ログファイル

システムの動作ログは`component_counter.log`に記録されます。問題が発生した際の診断に使用できます。

## 技術仕様

### 使用アルゴリズム

1. **輪郭検出**
   - OpenCVの適応的閾値処理
   - モルフォロジー処理によるノイズ除去
   - 輪郭面積とアスペクト比による部品フィルタリング

2. **テンプレートマッチング**
   - ハフ変換による円形部品検出
   - エッジ検出による形状認識

3. **YOLOv8**
   - 事前学習済みモデルによる物体検出
   - 信頼度ベースのフィルタリング
   - NMS（Non-Maximum Suppression）による重複除去

### 画像補正機能

- **基本補正**
  - 明度調整: ImageEnhanceライブラリによる高品質調整
  - コントラスト調整: 適応的なコントラスト強化
  - ガンマ補正: LUTテーブルによる高速処理

- **高度な補正**
  - CLAHE: 適応ヒストグラム均一化による局所コントラスト改善
  - シャープネス強化: カーネルフィルタによるエッジ強調
  - ノイズ除去: Non-Local Meansによる高品質ノイズ除去

### 品質保証メカニズム

- 複数アルゴリズムの結果比較
- 変動係数（CV）による品質スコア算出
- 統計的な異常値検出
- 画像補正による前処理品質向上

## サポート

### 開発者向け情報

- Python 3.8+対応
- OpenCV 4.8.1使用
- YOLOv8による物体検出
- tkinterによるGUI実装

### 拡張可能性

- 新しい検出アルゴリズムの追加
- 部品固有のテンプレート作成
- バッチ処理機能の実装
- Webインターフェースの追加

## ライセンス

このソフトウェアは工場内での使用を目的として開発されました。適切な品質管理の下で使用してください。

## 更新履歴

- v1.1.0: 画像補正機能追加
  - リアルタイム画像補正機能
  - 明度・コントラスト・ガンマ調整
  - CLAHE、シャープネス、ノイズ除去
  - 悪条件下での検出精度向上

- v1.0.0: 初期リリース
  - 基本的な部品カウント機能
  - 複数アルゴリズム対応
  - GUI実装 