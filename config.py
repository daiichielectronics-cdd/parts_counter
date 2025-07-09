"""
電子部品カウントシステム設定ファイル
"""

# アプリケーション設定
APP_TITLE = "電子部品カウントシステム"
APP_VERSION = "1.1.0"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# 画像処理設定
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
MAX_IMAGE_SIZE = (1024, 1024)  # 処理効率化のための最大サイズ

# 検出設定
CONFIDENCE_THRESHOLD = 0.6  # 信頼度閾値を上げて誤検出を減らす
NMS_THRESHOLD = 0.3  # NMS閾値を下げて重複除去を強化
MIN_COMPONENT_SIZE = 50  # 最小部品サイズを大きくしてノイズを除去

# 新しい誤検出対策パラメータ
MAX_COMPONENT_SIZE = 10000  # 最大部品サイズ制限
TEXTURE_FILTER_ENABLED = True  # テクスチャフィルタの有効化
NOISE_REDUCTION_LEVEL = 2  # ノイズ除去レベル（1-3）

# 形状フィルタリング設定
MIN_CIRCULARITY = 0.3  # 最小円形度
MIN_SOLIDITY = 0.7    # 最小充填率
MAX_ASPECT_RATIO = 5.0  # 最大アスペクト比
MIN_FILL_RATIO = 0.4   # 最小充填率
MAX_FILL_RATIO = 0.95  # 最大充填率

# YOLOv8設定
YOLO_MODEL = "yolov8n.pt"  # 軽量モデル（工場環境での高速処理）
YOLO_DEVICE = "cpu"  # CPUで実行（GPU非依存）

# 品質保証設定
ENABLE_MULTI_ALGORITHM = True  # 複数アルゴリズムの併用
ALGORITHM_AGREEMENT_THRESHOLD = 0.8  # アルゴリズム間一致率閾値

# ログ設定
LOG_LEVEL = "INFO"
LOG_FILE = "component_counter.log"

# UI設定
PREVIEW_SIZE = (400, 300)  # プレビュー画像サイズ
RESULT_FONT_SIZE = 12 