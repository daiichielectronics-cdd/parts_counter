#!/usr/bin/env python3
"""
電子部品カウントシステム セットアップスクリプト
"""

import os
import sys
import subprocess
import platform
import venv
from pathlib import Path

def print_header(title):
    """セクションヘッダーを表示"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def run_command(command, description, check=True):
    """コマンドを実行"""
    print(f"\n{description}...")
    print(f"実行: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"出力: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"エラー: {e}")
        if e.stderr:
            print(f"エラー詳細: {e.stderr}")
        return False

def check_python_version():
    """Python バージョンをチェック"""
    print_header("Python環境チェック")
    
    version = sys.version_info
    print(f"Python バージョン: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8以上が必要です")
        print("Python 3.8以上をインストールしてください")
        return False
    
    print("✅ Python バージョン要件を満たしています")
    return True

def create_virtual_environment():
    """仮想環境を作成"""
    print_header("仮想環境の作成")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("既存の仮想環境を削除中...")
        if platform.system() == "Windows":
            run_command("rmdir /s /q venv", "仮想環境削除", check=False)
        else:
            run_command("rm -rf venv", "仮想環境削除", check=False)
    
    print("新しい仮想環境を作成中...")
    venv.create("venv", with_pip=True)
    
    print("✅ 仮想環境を作成しました")
    return True

def install_dependencies():
    """依存関係をインストール"""
    print_header("依存関係のインストール")
    
    # 仮想環境のPythonパスを設定
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python.exe"
        pip_path = "venv\\Scripts\\pip.exe"
    else:
        python_path = "venv/bin/python"
        pip_path = "venv/bin/pip"
    
    # pipのアップグレード
    if not run_command(f"{pip_path} install --upgrade pip", "pipのアップグレード"):
        return False
    
    # 依存関係のインストール
    if not run_command(f"{pip_path} install -r requirements.txt", "依存関係のインストール"):
        return False
    
    print("✅ 依存関係のインストールが完了しました")
    return True

def download_yolo_model():
    """YOLOv8モデルをダウンロード"""
    print_header("YOLOv8モデルのダウンロード")
    
    # 仮想環境のPythonパスを設定
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python.exe"
    else:
        python_path = "venv/bin/python"
    
    # YOLOモデルのダウンロード
    download_command = f'{python_path} -c "from ultralytics import YOLO; YOLO(\'yolov8n.pt\')"'
    
    if not run_command(download_command, "YOLOv8モデルのダウンロード"):
        print("⚠️  YOLOモデルのダウンロードに失敗しました")
        print("   手動でダウンロードしてください:")
        print(f"   {python_path} -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
        return False
    
    print("✅ YOLOv8モデルのダウンロードが完了しました")
    return True

def create_test_image():
    """テスト用の画像を生成"""
    print_header("テスト用画像の生成")
    
    # 仮想環境のPythonパスを設定
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python.exe"
    else:
        python_path = "venv/bin/python"
    
    # テスト画像生成スクリプト
    test_script = '''
import numpy as np
import cv2
import os

# テスト用の電子部品画像を生成
def create_test_image():
    # 白い背景
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 電子部品っぽい矩形を描画
    components = [
        (50, 50, 30, 15),   # 抵抗1
        (100, 80, 25, 12),  # 抵抗2
        (200, 60, 35, 18),  # 抵抗3
        (300, 100, 28, 14), # 抵抗4
        (400, 70, 32, 16),  # 抵抗5
        (150, 150, 20, 20), # チップ1
        (250, 180, 18, 18), # チップ2
        (350, 160, 22, 22), # チップ3
        (80, 200, 40, 25),  # トランジスタ1
        (180, 220, 38, 23), # トランジスタ2
        (280, 200, 42, 27), # トランジスタ3
        (450, 180, 35, 20), # トランジスタ4
    ]
    
    # 部品を描画
    for i, (x, y, w, h) in enumerate(components):
        # 色を変える（グレー系）
        color = (80 + i*10, 80 + i*10, 80 + i*10)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 1)
    
    # ノイズを追加
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # テスト画像を保存
    os.makedirs("test_images", exist_ok=True)
    cv2.imwrite("test_images/test_components.jpg", img)
    print("テスト画像を生成しました: test_images/test_components.jpg")
    print(f"部品数: {len(components)}個")

if __name__ == "__main__":
    create_test_image()
'''
    
    # 一時的なスクリプトファイルを作成
    with open("generate_test_image.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    # テスト画像を生成
    success = run_command(f"{python_path} generate_test_image.py", "テスト画像の生成")
    
    # 一時ファイルを削除
    os.remove("generate_test_image.py")
    
    if success:
        print("✅ テスト用画像を生成しました")
        return True
    else:
        print("⚠️  テスト画像の生成に失敗しました")
        return False

def create_startup_script():
    """起動スクリプトを作成"""
    print_header("起動スクリプトの作成")
    
    if platform.system() == "Windows":
        # Windows用バッチファイル
        script_content = '''@echo off
echo 電子部品カウントシステムを起動中...
venv\\Scripts\\python.exe main.py
pause
'''
        with open("run_counter.bat", "w", encoding="utf-8") as f:
            f.write(script_content)
        print("✅ 起動スクリプトを作成しました: run_counter.bat")
        
    else:
        # macOS/Linux用fishシェルスクリプト
        script_content = '''#!/usr/bin/env fish
echo "電子部品カウントシステムを起動中..."
cd (dirname (status --current-filename))
./venv/bin/python main.py
'''
        with open("run_counter.fish", "w", encoding="utf-8") as f:
            f.write(script_content)
        
        # 実行権限を付与
        os.chmod("run_counter.fish", 0o755)
        print("✅ 起動スクリプトを作成しました: run_counter.fish")

def run_test():
    """アプリケーションのテスト実行"""
    print_header("アプリケーションのテスト")
    
    # 仮想環境のPythonパスを設定
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python.exe"
    else:
        python_path = "venv/bin/python"
    
    # インポートテスト
    test_command = f'{python_path} -c "import config; import image_processor; print(\'✅ モジュールのインポートが成功しました\')"'
    
    if run_command(test_command, "モジュールのインポートテスト"):
        print("✅ セットアップが正常に完了しました")
        return True
    else:
        print("❌ セットアップに問題があります")
        return False

def main():
    """メイン処理"""
    print_header("電子部品カウントシステム セットアップ")
    print("このスクリプトは以下の処理を実行します:")
    print("1. Python環境のチェック")
    print("2. 仮想環境の作成")
    print("3. 依存関係のインストール")
    print("4. YOLOv8モデルのダウンロード")
    print("5. テスト用画像の生成")
    print("6. 起動スクリプトの作成")
    print("7. アプリケーションのテスト")
    
    input("\nEnterキーを押して続行...")
    
    # セットアップ処理
    steps = [
        ("Python環境チェック", check_python_version),
        ("仮想環境作成", create_virtual_environment),
        ("依存関係インストール", install_dependencies),
        ("YOLOモデルダウンロード", download_yolo_model),
        ("テスト画像生成", create_test_image),
        ("起動スクリプト作成", create_startup_script),
        ("アプリケーションテスト", run_test),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        if not step_func():
            failed_steps.append(step_name)
    
    # 結果表示
    print_header("セットアップ結果")
    
    if not failed_steps:
        print("🎉 セットアップが完了しました！")
        print("\n次のステップ:")
        if platform.system() == "Windows":
            print("1. run_counter.bat をダブルクリックしてアプリケーションを起動")
        else:
            print("1. ./run_counter.fish を実行してアプリケーションを起動")
        print("2. test_images/test_components.jpg を使用してテスト")
        print("3. READMEファイルで詳細な使用方法を確認")
    else:
        print("❌ 以下のステップで問題が発生しました:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nREADMEファイルのトラブルシューティングセクションを参照してください")

if __name__ == "__main__":
    main() 