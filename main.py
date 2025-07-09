"""
電子部品カウントシステム - メインGUIアプリケーション
工場作業者向けの直感的なインターフェース
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import os
import threading
import time
from datetime import datetime
import json

import config
from image_processor import ComponentDetector

class ComponentCounterGUI:
    """電子部品カウントシステムGUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.detector = ComponentDetector()
        self.current_image_path = None
        self.current_result = None
        self.processing = False
        
        # 画像補正用変数
        self.original_image = None
        self.processed_image = None
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.enable_clahe_var = tk.BooleanVar(value=False)
        self.enable_sharpen_var = tk.BooleanVar(value=False)
        self.enable_denoise_var = tk.BooleanVar(value=False)
        
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        """UIの初期設定"""
        self.root.title(f"{config.APP_TITLE} v{config.APP_VERSION}")
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.resizable(True, True)
        
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # グリッド設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)  # 画像表示パネルが3列目になる
        main_frame.rowconfigure(1, weight=1)
        
        # タイトル
        title_label = ttk.Label(main_frame, text=config.APP_TITLE, 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 左側パネル（コントロール）
        self.setup_control_panel(main_frame)
        
        # 中央左パネル（画像補正）
        self.setup_enhancement_panel(main_frame)
        
        # 右側パネル（画像表示）
        self.setup_image_panel(main_frame)
        
        # 下部パネル（結果表示）
        self.setup_result_panel(main_frame)
        
        # ステータスバー
        self.setup_status_bar()
        
    def setup_control_panel(self, parent):
        """コントロールパネルの設定"""
        control_frame = ttk.LabelFrame(parent, text="操作パネル", padding="10")
        control_frame.grid(row=1, column=0, sticky="nsew", 
                          padx=(0, 10))
        
        # 画像選択ボタン
        self.select_button = ttk.Button(control_frame, text="画像を選択", 
                                       command=self.select_image,
                                       style="Accent.TButton")
        self.select_button.grid(row=0, column=0, sticky="we", pady=(0, 10))
        
        # 処理開始ボタン
        self.process_button = ttk.Button(control_frame, text="部品カウント実行", 
                                        command=self.start_processing,
                                        state=tk.DISABLED)
        self.process_button.grid(row=1, column=0, sticky="we", pady=(0, 10))
        
        # プログレスバー
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, sticky="we", pady=(0, 10))
        
        # 設定フレーム
        settings_frame = ttk.LabelFrame(control_frame, text="設定", padding="5")
        settings_frame.grid(row=3, column=0, sticky="we", pady=(0, 10))
        
        # 信頼度閾値
        ttk.Label(settings_frame, text="信頼度閾値:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=config.CONFIDENCE_THRESHOLD)
        confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                    variable=self.confidence_var, 
                                    orient=tk.HORIZONTAL, length=120)
        confidence_scale.grid(row=0, column=1, sticky="we", padx=(5, 0))
        
        # 検出アルゴリズム選択
        ttk.Label(settings_frame, text="検出方法:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.algorithm_var = tk.StringVar(value="all")
        algorithm_combo = ttk.Combobox(settings_frame, textvariable=self.algorithm_var,
                                     values=["all", "contour", "template", "yolo"],
                                     state="readonly", width=15)
        algorithm_combo.grid(row=1, column=1, sticky="we", padx=(5, 0), pady=(5, 0))
        
        # 結果保存ボタン
        self.save_button = ttk.Button(control_frame, text="結果を保存", 
                                     command=self.save_results,
                                     state=tk.DISABLED)
        self.save_button.grid(row=4, column=0, sticky="we", pady=(10, 0))
        
        # カラム設定
        control_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=1)
        
    def setup_enhancement_panel(self, parent):
        """画像補正パネルの設定"""
        enhancement_frame = ttk.LabelFrame(parent, text="画像補正", padding="10")
        enhancement_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 10))
        
        # 明度調整
        ttk.Label(enhancement_frame, text="明度:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        brightness_scale = ttk.Scale(enhancement_frame, from_=0.5, to=2.0, 
                                    variable=self.brightness_var, orient="horizontal", 
                                    length=150, command=self.on_enhancement_change)
        brightness_scale.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 5))
        
        brightness_label = ttk.Label(enhancement_frame, text="1.00", width=4)
        brightness_label.grid(row=0, column=2, sticky="w", padx=(5, 0), pady=(0, 5))
        
        # コントラスト調整
        ttk.Label(enhancement_frame, text="コントラスト:").grid(row=1, column=0, sticky="w", pady=(0, 5))
        contrast_scale = ttk.Scale(enhancement_frame, from_=0.5, to=3.0,
                                  variable=self.contrast_var, orient="horizontal",
                                  length=150, command=self.on_enhancement_change)
        contrast_scale.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(0, 5))
        
        contrast_label = ttk.Label(enhancement_frame, text="1.00", width=4)
        contrast_label.grid(row=1, column=2, sticky="w", padx=(5, 0), pady=(0, 5))
        
        # ガンマ補正
        ttk.Label(enhancement_frame, text="ガンマ:").grid(row=2, column=0, sticky="w", pady=(0, 5))
        gamma_scale = ttk.Scale(enhancement_frame, from_=0.5, to=2.0,
                               variable=self.gamma_var, orient="horizontal",
                               length=150, command=self.on_enhancement_change)
        gamma_scale.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=(0, 5))
        
        gamma_label = ttk.Label(enhancement_frame, text="1.00", width=4)
        gamma_label.grid(row=2, column=2, sticky="w", padx=(5, 0), pady=(0, 5))
        
        # 高度な補正オプション
        ttk.Label(enhancement_frame, text="高度な補正:").grid(row=3, column=0, columnspan=3, 
                                                     sticky="w", pady=(10, 5))
        
        # CLAHE（適応ヒストグラム均一化）
        clahe_check = ttk.Checkbutton(enhancement_frame, text="ヒストグラム均一化",
                                     variable=self.enable_clahe_var,
                                     command=self.on_enhancement_change)
        clahe_check.grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 2))
        
        # シャープネス強化
        sharpen_check = ttk.Checkbutton(enhancement_frame, text="シャープネス強化",
                                       variable=self.enable_sharpen_var,
                                       command=self.on_enhancement_change)
        sharpen_check.grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 2))
        
        # ノイズ除去
        denoise_check = ttk.Checkbutton(enhancement_frame, text="ノイズ除去",
                                       variable=self.enable_denoise_var,
                                       command=self.on_enhancement_change)
        denoise_check.grid(row=6, column=0, columnspan=3, sticky="w", pady=(0, 2))
        
        # 誤検出対策オプション
        ttk.Label(enhancement_frame, text="誤検出対策:").grid(row=7, column=0, columnspan=3, 
                                                      sticky="w", pady=(10, 5))
        
        # テクスチャフィルタ
        self.enable_texture_filter_var = tk.BooleanVar(value=True)
        texture_check = ttk.Checkbutton(enhancement_frame, text="テクスチャフィルタ",
                                       variable=self.enable_texture_filter_var)
        texture_check.grid(row=8, column=0, columnspan=3, sticky="w", pady=(0, 2))
        
        # ノイズ除去レベル
        ttk.Label(enhancement_frame, text="ノイズ除去レベル:").grid(row=9, column=0, sticky="w", pady=(5, 0))
        self.noise_level_var = tk.IntVar(value=2)
        noise_level_scale = ttk.Scale(enhancement_frame, from_=1, to=3,
                                     variable=self.noise_level_var, orient="horizontal",
                                     length=150)
        noise_level_scale.grid(row=9, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        
        noise_level_label = ttk.Label(enhancement_frame, text="2", width=4)
        noise_level_label.grid(row=9, column=2, sticky="w", padx=(5, 0), pady=(5, 0))
        
        # 最小部品サイズ
        ttk.Label(enhancement_frame, text="最小部品サイズ:").grid(row=10, column=0, sticky="w", pady=(5, 0))
        self.min_size_var = tk.IntVar(value=50)
        min_size_scale = ttk.Scale(enhancement_frame, from_=10, to=200,
                                  variable=self.min_size_var, orient="horizontal",
                                  length=150)
        min_size_scale.grid(row=10, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        
        min_size_label = ttk.Label(enhancement_frame, text="50", width=4)
        min_size_label.grid(row=10, column=2, sticky="w", padx=(5, 0), pady=(5, 0))
        
        # リセットボタン
        reset_button = ttk.Button(enhancement_frame, text="リセット",
                                 command=self.reset_enhancement)
        reset_button.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        # グリッド設定
        enhancement_frame.columnconfigure(1, weight=1)

    def setup_image_panel(self, parent):
        """画像表示パネルの設定"""
        image_frame = ttk.LabelFrame(parent, text="画像プレビュー", padding="10")
        image_frame.grid(row=1, column=2, sticky="nsew")
        
        # 画像キャンバス
        self.canvas = tk.Canvas(image_frame, bg="white", width=400, height=300)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # スクロールバー
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="we")
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # 初期表示
        self.show_placeholder_image()
        
        # グリッド設定
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
    def setup_result_panel(self, parent):
        """結果表示パネルの設定"""
        result_frame = ttk.LabelFrame(parent, text="検出結果", padding="10")
        result_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", 
                         pady=(10, 0))
        
        # 結果表示エリア
        self.result_text = scrolledtext.ScrolledText(result_frame, height=8, width=80)
        self.result_text.grid(row=0, column=0, sticky="nsew")
        
        # グリッド設定
        result_frame.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=0)
        
    def setup_status_bar(self):
        """ステータスバーの設定"""
        self.status_var = tk.StringVar(value="準備完了")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky="we")
        
    def setup_styles(self):
        """スタイルの設定"""
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
        
    def show_placeholder_image(self):
        """プレースホルダー画像の表示"""
        # 空の画像を作成
        placeholder = Image.new('RGB', (400, 300), color='lightgray')
        self.photo = ImageTk.PhotoImage(placeholder)
        self.canvas.create_image(200, 150, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def select_image(self):
        """画像ファイルの選択"""
        file_types = [
            ("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("すべてのファイル", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="部品画像を選択してください",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.process_button.config(state=tk.NORMAL)
            self.status_var.set(f"画像を選択: {os.path.basename(file_path)}")
            
    def display_image(self, image_path):
        """画像の表示"""
        try:
            # 元画像をリセット
            self.original_image = None
            self.processed_image = None
            
            # 画像読み込み
            image = Image.open(image_path)
            
            # プレビューサイズにリサイズ
            image.thumbnail(config.PREVIEW_SIZE, Image.Resampling.LANCZOS)
            
            # tkinter用に変換
            self.photo = ImageTk.PhotoImage(image)
            
            # キャンバスに表示
            self.canvas.delete("all")
            self.canvas.create_image(
                config.PREVIEW_SIZE[0]//2, config.PREVIEW_SIZE[1]//2, 
                image=self.photo
            )
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("エラー", f"画像の表示に失敗しました: {str(e)}")
            
    def start_processing(self):
        """部品カウント処理の開始"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "画像を選択してください")
            return
            
        if self.processing:
            messagebox.showwarning("警告", "処理中です。しばらくお待ちください")
            return
            
        # UIの無効化
        self.process_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status_var.set("部品カウント処理中...")
        
        # 別スレッドで処理実行
        threading.Thread(target=self.process_image, daemon=True).start()
        
    def process_image(self):
        """画像処理の実行"""
        try:
            self.processing = True
            
            # 設定の更新
            config.CONFIDENCE_THRESHOLD = self.confidence_var.get()
            config.TEXTURE_FILTER_ENABLED = self.enable_texture_filter_var.get()
            config.NOISE_REDUCTION_LEVEL = self.noise_level_var.get()
            config.MIN_COMPONENT_SIZE = self.min_size_var.get()
            
            # 部品カウント実行（補正済み画像があれば使用）
            image_path = self.get_processed_image_path()
            if image_path is None:
                raise ValueError("画像パスが設定されていません")
            
            # 選択されたアルゴリズムを取得
            selected_algorithm = self.algorithm_var.get()
            
            result = self.detector.count_components(image_path, selected_algorithm)
            
            # UIの更新（メインスレッドで実行）
            self.root.after(0, self.update_results, result)
            
        except Exception as e:
            self.root.after(0, self.show_error, f"処理エラー: {str(e)}")
        finally:
            self.processing = False
            
    def update_results(self, result):
        """結果の更新"""
        self.current_result = result
        
        # プログレスバー停止
        self.progress.stop()
        
        # UIの再有効化
        self.process_button.config(state=tk.NORMAL)
        self.select_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        
        if 'error' in result:
            self.show_error(result['error'])
            return
            
        # 結果表示
        self.display_results(result)
        
        # 検出結果画像の表示
        if 'processed_image' in result and 'detections' in result:
            annotated_image = self.detector.draw_detections(
                result['processed_image'], result['detections']
            )
            self.display_annotated_image(annotated_image)
            
        self.status_var.set(f"処理完了: {result['count']}個の部品を検出")
        
    def display_results(self, result):
        """結果の詳細表示"""
        self.result_text.delete(1.0, tk.END)
        
        # 基本情報
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output = f"=== 部品カウント結果 ===\n"
        output += f"処理時刻: {timestamp}\n"
        output += f"画像ファイル: {os.path.basename(self.current_image_path or '')}\n"
        output += f"検出部品数: {result['count']}個\n"
        output += f"品質スコア: {result.get('quality_score', 0):.2f}\n\n"
        
        # アルゴリズム別結果
        if 'algorithm_results' in result:
            output += "=== アルゴリズム別結果 ===\n"
            for algo, count in result['algorithm_results'].items():
                output += f"  {algo}: {count}個\n"
            output += "\n"
            
        # 検出詳細
        if 'detections' in result:
            output += "=== 検出詳細 ===\n"
            for i, detection in enumerate(result['detections'], 1):
                bbox = detection['bbox']
                conf = detection['confidence']
                method = detection['method']
                output += f"  部品{i}: 位置({bbox[0]}, {bbox[1]}) "
                output += f"サイズ({bbox[2]}x{bbox[3]}) "
                output += f"信頼度{conf:.2f} 手法:{method}\n"
                
        # 品質評価
        quality_score = result.get('quality_score', 0)
        if quality_score < 0.5:
            output += "\n⚠️ 警告: 検出品質が低い可能性があります。\n"
            output += "異なる照明条件や撮影角度で再試行することを推奨します。\n"
        elif quality_score > 0.8:
            output += "\n✅ 検出品質は良好です。\n"
            
        self.result_text.insert(tk.END, output)
        
    def display_annotated_image(self, image):
        """注釈付き画像の表示"""
        try:
            # OpenCV (BGR) から PIL (RGB) に変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # プレビューサイズにリサイズ
            pil_image.thumbnail(config.PREVIEW_SIZE, Image.Resampling.LANCZOS)
            
            # tkinter用に変換
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # キャンバスに表示
            self.canvas.delete("all")
            self.canvas.create_image(
                config.PREVIEW_SIZE[0]//2, config.PREVIEW_SIZE[1]//2, 
                image=self.photo
            )
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            print(f"注釈付き画像表示エラー: {e}")
            
    def save_results(self):
        """結果の保存"""
        if not self.current_result:
            messagebox.showwarning("警告", "保存する結果がありません")
            return
            
        try:
            # 保存ファイル名の生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.current_image_path or ''))[0]
            filename = f"result_{base_name}_{timestamp}.json"
            
            # 保存先選択
            save_path = filedialog.asksaveasfilename(
                title="結果を保存",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=filename
            )
            
            if save_path:
                # 結果をJSON形式で保存
                save_data = {
                    'timestamp': datetime.now().isoformat(),
                    'image_path': self.current_image_path,
                    'count': self.current_result['count'],
                    'quality_score': self.current_result.get('quality_score', 0),
                    'algorithm_results': self.current_result.get('algorithm_results', {}),
                    'detections': self.current_result.get('detections', [])
                }
                
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                    
                messagebox.showinfo("保存完了", f"結果を保存しました: {save_path}")
                self.status_var.set(f"結果を保存: {os.path.basename(save_path)}")
                
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました: {str(e)}")
            
    def on_enhancement_change(self, event=None):
        """画像補正パラメータが変更されたときの処理"""
        if self.current_image_path and self.original_image is None:
            # 元画像を保存
            self.original_image = cv2.imread(self.current_image_path)
        
        if self.original_image is not None:
            # 補正された画像を表示
            self.apply_enhancement_and_display()
    
    def apply_enhancement_and_display(self):
        """画像補正を適用して表示"""
        try:
            if self.original_image is None:
                return
            
            # OpenCV画像をPILに変換
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 明度調整
            brightness = self.brightness_var.get()
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness)
            
            # コントラスト調整
            contrast = self.contrast_var.get()
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast)
            
            # PIL画像を再びOpenCVに変換
            enhanced_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # ガンマ補正
            gamma = self.gamma_var.get()
            if gamma != 1.0:
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                                 for i in range(256)]).astype("uint8")
                enhanced_cv = cv2.LUT(enhanced_cv, table)
            
            # CLAHE（適応ヒストグラム均一化）
            if self.enable_clahe_var.get():
                lab = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced_cv = cv2.merge([l, a, b])
                enhanced_cv = cv2.cvtColor(enhanced_cv, cv2.COLOR_LAB2BGR)
            
            # シャープネス強化
            if self.enable_sharpen_var.get():
                kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
                enhanced_cv = cv2.filter2D(enhanced_cv, -1, kernel)
            
            # ノイズ除去
            if self.enable_denoise_var.get():
                enhanced_cv = cv2.fastNlMeansDenoisingColored(enhanced_cv, None, 10, 10, 7, 21)
            
            # 処理済み画像を保存
            self.processed_image = enhanced_cv
            
            # 表示用に変換
            image_rgb = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # プレビューサイズにリサイズ
            pil_image.thumbnail(config.PREVIEW_SIZE, Image.Resampling.LANCZOS)
            
            # tkinter用に変換
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # キャンバスに表示
            self.canvas.delete("all")
            self.canvas.create_image(
                config.PREVIEW_SIZE[0]//2, config.PREVIEW_SIZE[1]//2, 
                image=self.photo
            )
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            print(f"画像補正エラー: {e}")
    
    def reset_enhancement(self):
        """画像補正パラメータをリセット"""
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.gamma_var.set(1.0)
        self.enable_clahe_var.set(False)
        self.enable_sharpen_var.set(False)
        self.enable_denoise_var.set(False)
        
        # 誤検出対策設定もリセット
        self.enable_texture_filter_var.set(True)
        self.noise_level_var.set(2)
        self.min_size_var.set(50)
        
        # 元画像を再表示
        if self.original_image is not None:
            self.processed_image = None
            if self.current_image_path:
                self.display_image(self.current_image_path)
    
    def get_processed_image_path(self):
        """処理用画像のパスを取得（補正済みがあれば一時ファイルを作成）"""
        if self.processed_image is not None:
            # 一時ファイルに補正済み画像を保存
            temp_path = "temp_enhanced_image.jpg"
            cv2.imwrite(temp_path, self.processed_image)
            return temp_path
        return self.current_image_path

    def show_error(self, message):
        """エラーメッセージの表示"""
        messagebox.showerror("エラー", message)
        self.progress.stop()
        self.process_button.config(state=tk.NORMAL)
        self.select_button.config(state=tk.NORMAL)
        self.status_var.set("エラーが発生しました")
        
    def run(self):
        """アプリケーションの実行"""
        self.root.mainloop()

def main():
    """メイン関数"""
    try:
        app = ComponentCounterGUI()
        app.run()
    except Exception as e:
        print(f"アプリケーション起動エラー: {e}")

if __name__ == "__main__":
    main() 