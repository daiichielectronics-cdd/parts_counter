"""
電子部品画像処理モジュール
高精度な部品検出のための複数アルゴリズムを実装
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import List, Tuple, Dict, Any
import os
from ultralytics import YOLO
import config

# ログ設定
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComponentDetector:
    """電子部品検出クラス"""
    
    def __init__(self):
        self.yolo_model = None
        self.initialize_models()
        
    def initialize_models(self):
        """AIモデルの初期化"""
        try:
            logger.info("YOLOv8モデルを初期化中...")
            self.yolo_model = YOLO(config.YOLO_MODEL)
            logger.info("モデル初期化完了")
        except Exception as e:
            logger.error(f"モデル初期化エラー: {e}")
            self.yolo_model = None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """画像の前処理"""
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"画像を読み込めません: {image_path}")
            
            # サイズ調整
            height, width = image.shape[:2]
            if width > config.MAX_IMAGE_SIZE[0] or height > config.MAX_IMAGE_SIZE[1]:
                scale = min(config.MAX_IMAGE_SIZE[0]/width, config.MAX_IMAGE_SIZE[1]/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            logger.info(f"画像サイズ: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"画像前処理エラー: {e}")
            raise
    
    def detect_with_contours(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """輪郭検出による部品カウント（誤検出対策強化版）"""
        try:
            # グレースケール変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 強力なノイズ除去（JPEGブロックノイズ対策）
            # バイラテラルフィルタでエッジを保持しながらノイズ除去
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # さらにガウシアンブラーでテクスチャを平滑化
            blurred = cv2.GaussianBlur(denoised, (7, 7), 0)
            
            # 適応的閾値処理（パラメータ調整）
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 3  # ブロックサイズとC値を調整
            )
            
            # 強化されたモルフォロジー処理（ノイズ除去）
            kernel_small = np.ones((2, 2), np.uint8)
            kernel_medium = np.ones((3, 3), np.uint8)
            kernel_large = np.ones((5, 5), np.uint8)
            
            # 小さなノイズを除去
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
            # 隙間を埋める
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
            # 大きなノイズを除去
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_large)
            
            # 輪郭検出
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 部品候補をフィルタリング（厳密な条件）
            components = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 面積フィルタリング（より厳密）
                if area < config.MIN_COMPONENT_SIZE * 2 or area > config.MIN_COMPONENT_SIZE * 500:
                    continue
                
                # バウンディングボックスの取得
                x, y, w, h = cv2.boundingRect(contour)
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # サイズフィルタリング
                if w < 8 or h < 8 or w > 200 or h > 200:
                    continue
                
                # アスペクト比チェック（電子部品の一般的な形状）
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio > 5.0:  # 細長すぎる形状を除外
                    continue
                
                # 輪郭の複雑さ分析（ノイズ除去）
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                # 円形度チェック（4πA/P²）
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.3:  # 複雑すぎる形状を除外
                    continue
                
                # 充填率チェック（実際の面積/バウンディングボックス面積）
                bbox_area = w * h
                if bbox_area == 0:
                    continue
                
                fill_ratio = area / bbox_area
                if fill_ratio < 0.4 or fill_ratio > 0.95:  # 不自然な充填率を除外
                    continue
                
                # 輪郭の凸包解析
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                
                solidity = area / hull_area
                if solidity < 0.7:  # 凹凸が多すぎる形状を除外
                    continue
                
                # 信頼度の計算（複数の要素を考慮）
                confidence = 0.5  # ベース信頼度
                
                # 良い形状ほど信頼度が高い
                if 0.7 <= circularity <= 1.0:
                    confidence += 0.2
                if 0.7 <= fill_ratio <= 0.9:
                    confidence += 0.1
                if solidity > 0.9:
                    confidence += 0.1
                if 1.0 <= aspect_ratio <= 3.0:
                    confidence += 0.1
                
                components.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': min(confidence, 1.0),
                    'method': 'contour',
                    'circularity': circularity,
                    'fill_ratio': fill_ratio,
                    'solidity': solidity
                })
            
            logger.info(f"輪郭検出結果: {len(components)}個の部品候補")
            return components
            
        except Exception as e:
            logger.error(f"輪郭検出エラー: {e}")
            return []
    
    def detect_with_template_matching(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """テンプレートマッチングによる部品検出（誤検出対策強化版）"""
        try:
            # グレースケール変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 強力なノイズ除去（テクスチャ対策）
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
            
            # エッジ検出（パラメータ調整）
            edges = cv2.Canny(blurred, 30, 100)  # 閾値を調整
            
            # ハフ変換による円形部品検出（パラメータ厳密化）
            circles = cv2.HoughCircles(
                blurred,  # エッジではなく平滑化画像を使用
                cv2.HOUGH_GRADIENT,
                dp=1.2,        # 解像度比率を調整
                minDist=30,    # 円同士の最小距離を大きく
                param1=50,     # Canny高閾値
                param2=40,     # 蓄積閾値を高く（偽陽性を減らす）
                minRadius=8,   # 最小半径を大きく
                maxRadius=80   # 最大半径を制限
            )
            
            components = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # 境界チェック
                    if x - r < 0 or y - r < 0 or x + r >= gray.shape[1] or y + r >= gray.shape[0]:
                        continue
                    
                    # 円の品質チェック
                    roi = gray[y-r:y+r, x-r:x+r]
                    if roi.size == 0:
                        continue
                    
                    # ROI内の標準偏差チェック（テクスチャ除去）
                    roi_std = np.std(roi)
                    if roi_std < 10:  # 変化が少なすぎる（平坦すぎる）
                        continue
                    if roi_std > 60:  # 変化が多すぎる（ノイズ）
                        continue
                    
                    # 円の輪郭の明瞭さチェック
                    # 円周上の点でエッジ強度を測定
                    edge_strength = 0
                    edge_count = 0
                    for angle in np.linspace(0, 2*np.pi, 16):
                        edge_x = int(x + r * np.cos(angle))
                        edge_y = int(y + r * np.sin(angle))
                        if 0 <= edge_x < edges.shape[1] and 0 <= edge_y < edges.shape[0]:
                            if edges[edge_y, edge_x] > 0:
                                edge_strength += 1
                            edge_count += 1
                    
                    # エッジ強度が不十分な場合は除外
                    if edge_count > 0:
                        edge_ratio = edge_strength / edge_count
                        if edge_ratio < 0.3:  # 円周の30%以上にエッジが必要
                            continue
                    
                    # 半径による信頼度調整
                    confidence = 0.6  # ベース信頼度
                    
                    # 適切なサイズの部品ほど信頼度が高い
                    if 10 <= r <= 30:
                        confidence += 0.2
                    elif 8 <= r <= 40:
                        confidence += 0.1
                    
                    # ROIの品質による信頼度調整
                    if 20 <= roi_std <= 40:
                        confidence += 0.1
                    
                    # エッジ明瞭度による信頼度調整
                    if edge_count > 0 and edge_ratio > 0.5:
                        confidence += 0.1
                    
                    components.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'area': np.pi * r * r,
                        'confidence': min(confidence, 1.0),
                        'method': 'template',
                        'radius': r,
                        'roi_std': roi_std,
                        'edge_ratio': edge_ratio if edge_count > 0 else 0
                    })
            
            # 重複する円を除去（距離ベース）
            if len(components) > 1:
                filtered_components = []
                for i, comp1 in enumerate(components):
                    is_duplicate = False
                    x1, y1, w1, h1 = comp1['bbox']
                    center1 = (x1 + w1//2, y1 + h1//2)
                    
                    for j, comp2 in enumerate(filtered_components):
                        x2, y2, w2, h2 = comp2['bbox']
                        center2 = (x2 + w2//2, y2 + h2//2)
                        
                        # 中心間距離
                        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                        min_radius = min(w1, w2) // 2
                        
                        if distance < min_radius * 1.5:  # 重複判定
                            # 信頼度の高い方を残す
                            if comp1['confidence'] <= comp2['confidence']:
                                is_duplicate = True
                                break
                            else:
                                filtered_components[j] = comp1
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        filtered_components.append(comp1)
                
                components = filtered_components
            
            logger.info(f"テンプレートマッチング結果: {len(components)}個の部品候補")
            return components
            
        except Exception as e:
            logger.error(f"テンプレートマッチングエラー: {e}")
            return []
    
    def detect_with_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """YOLOv8による部品検出"""
        try:
            if self.yolo_model is None:
                logger.warning("YOLOモデルが利用できません")
                return []
            
            # YOLO推論
            results = self.yolo_model(image, conf=config.CONFIDENCE_THRESHOLD)
            
            components = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        components.append({
                            'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                            'area': (x2-x1) * (y2-y1),
                            'confidence': float(conf),
                            'method': 'yolo'
                        })
            
            logger.info(f"YOLO検出結果: {len(components)}個の部品候補")
            return components
            
        except Exception as e:
            logger.error(f"YOLO検出エラー: {e}")
            return []
    
    def merge_detections(self, detections_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """複数の検出結果をマージ"""
        try:
            all_detections = []
            for detections in detections_list:
                all_detections.extend(detections)
            
            if not all_detections:
                return []
            
            # NMS（Non-Maximum Suppression）による重複除去
            boxes = []
            scores = []
            
            for detection in all_detections:
                x, y, w, h = detection['bbox']
                boxes.append([x, y, x+w, y+h])
                scores.append(detection['confidence'])
            
            if boxes:
                boxes = np.array(boxes)
                scores = np.array(scores)
                
                # OpenCVのNMSを使用
                indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(), scores.tolist(), 
                    config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD
                )
                
                merged_detections = []
                if len(indices) > 0:
                    indices_array = np.array(indices).flatten()
                    for i in indices_array:
                        merged_detections.append(all_detections[i])
                
                logger.info(f"マージ後の検出結果: {len(merged_detections)}個")
                return merged_detections
            
            return []
            
        except Exception as e:
            logger.error(f"検出結果マージエラー: {e}")
            return []
    
    def count_components(self, image_path: str, algorithm: str = "all") -> Dict[str, Any]:
        """部品カウントのメイン処理"""
        try:
            logger.info(f"部品カウント開始: {image_path} (アルゴリズム: {algorithm})")
            
            # 画像前処理
            image = self.preprocess_image(image_path)
            
            # アルゴリズム選択による検出
            detections_list = []
            algorithm_results = {}
            
            if algorithm == "all" or algorithm == "contour":
                # 輪郭検出
                contour_detections = self.detect_with_contours(image)
                detections_list.append(contour_detections)
                algorithm_results['contour'] = len(contour_detections)
            else:
                algorithm_results['contour'] = 0
            
            if algorithm == "all" or algorithm == "template":
                # テンプレートマッチング
                template_detections = self.detect_with_template_matching(image)
                detections_list.append(template_detections)
                algorithm_results['template'] = len(template_detections)
            else:
                algorithm_results['template'] = 0
            
            if algorithm == "all" or algorithm == "yolo":
                # YOLO検出
                yolo_detections = self.detect_with_yolo(image)
                detections_list.append(yolo_detections)
                algorithm_results['yolo'] = len(yolo_detections)
            else:
                algorithm_results['yolo'] = 0
            
            # 結果マージ
            final_detections = self.merge_detections(detections_list)
            
            # 品質チェック
            quality_score = self.calculate_quality_score(detections_list)
            
            result = {
                'count': len(final_detections),
                'detections': final_detections,
                'quality_score': quality_score,
                'algorithm_results': algorithm_results,
                'processed_image': image,
                'selected_algorithm': algorithm
            }
            
            logger.info(f"部品カウント完了: {result['count']}個検出 (品質スコア: {quality_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"部品カウントエラー: {e}")
            return {'count': 0, 'error': str(e)}
    
    def calculate_quality_score(self, detections_list: List[List[Dict[str, Any]]]) -> float:
        """検出品質スコアの計算"""
        try:
            counts = [len(detections) for detections in detections_list if detections]
            
            if not counts:
                return 0.0
            
            # 標準偏差による一致度評価
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            
            if mean_count == 0:
                return 0.0
            
            # 変動係数（CV）を用いた品質スコア
            cv = std_count / mean_count
            quality_score = max(0.0, 1.0 - cv)
            
            logger.info(f"品質スコア: {quality_score:.2f}")
            return quality_score
            
        except Exception as e:
            logger.error(f"品質スコア計算エラー: {e}")
            return 0.0
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """検出結果を画像に描画"""
        try:
            result_image = image.copy()
            
            for detection in detections:
                x, y, w, h = detection['bbox']
                confidence = detection['confidence']
                method = detection['method']
                
                # 信頼度によって色を変更
                if confidence > 0.8:
                    color = (0, 255, 0)  # 緑: 高信頼度
                elif confidence > 0.5:
                    color = (0, 255, 255)  # 黄: 中信頼度
                else:
                    color = (0, 0, 255)  # 赤: 低信頼度
                
                # バウンディングボックス描画
                cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
                
                # ラベル描画
                label = f"{method}: {confidence:.2f}"
                cv2.putText(result_image, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return result_image
            
        except Exception as e:
            logger.error(f"描画エラー: {e}")
            return image 