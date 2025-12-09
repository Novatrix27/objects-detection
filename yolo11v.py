"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    YOLO11 ULTIMATE DETECTION SYSTEM                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Fonctionnalités:
• Auto-détection du matériel (GPU/CPU)
• Modèle le plus puissant adaptatif
• Logs ultra-détaillés en français
• Statistiques en temps réel
• Optimisation automatique
• Support multi-GPU
• Catégories personnalisées
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from pathlib import Path
import time
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Set
import json
import sys
import platform
from dataclasses import dataclass, asdict

# ============================================================================
#                         CONFIGURATION DES LOGS
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour terminal"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Vert
        'WARNING': '\033[33m',    # Jaune
        'ERROR': '\033[31m',      # Rouge
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

# Configuration du logging avec couleurs
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter(
    '%(levelname)s | %(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
))

file_handler = logging.FileHandler('yolo11_detection.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(
    '%(levelname)s | %(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)


# ============================================================================
#                         DÉTECTION DU MATÉRIEL
# ============================================================================

def detect_hardware():
    """Détecte automatiquement le meilleur matériel disponible"""
    logger.info("=" * 80)
    logger.info("🔍 DÉTECTION DU MATÉRIEL")
    logger.info("=" * 80)
    
    # Informations système
    logger.info(f"💻 Système: {platform.system()} {platform.release()}")
    logger.info(f"🖥️  Processeur: {platform.processor()}")
    logger.info(f"🐍 Python: {sys.version.split()[0]}")
    
    # Détection GPU NVIDIA (CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"✅ GPU NVIDIA Détecté!")
            logger.info(f"   • Nom: {gpu_name}")
            logger.info(f"   • Nombre de GPUs: {gpu_count}")
            logger.info(f"   • Mémoire: {gpu_memory:.1f} GB")
            logger.info(f"   • CUDA Version: {torch.version.cuda}")
            
            # Choisir le modèle selon la mémoire GPU
            if gpu_memory >= 8:
                model = "yolo11x.pt"  # Le plus puissant
                logger.info(f"   • Modèle recommandé: YOLO11x (Maximum de précision)")
            elif gpu_memory >= 6:
                model = "yolo11l.pt"  # Large
                logger.info(f"   • Modèle recommandé: YOLO11l (Haute précision)")
            else:
                model = "yolo11m.pt"  # Medium
                logger.info(f"   • Modèle recommandé: YOLO11m (Équilibré)")
            
            return "cuda", model, True, gpu_memory
            
    except ImportError:
        logger.warning("⚠️  PyTorch non installé - impossible de détecter CUDA")
    except Exception as e:
        logger.warning(f"⚠️  Erreur détection CUDA: {e}")
    
    # Détection GPU Apple (Metal)
    if platform.system() == "Darwin":  # macOS
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("✅ GPU Apple Silicon Détecté (M1/M2/M3)!")
                logger.info("   • Metal Performance Shaders activé")
                logger.info("   • Modèle recommandé: YOLO11l")
                return "mps", "yolo11l.pt", False, 0
        except:
            pass
    
    # Mode CPU par défaut
    logger.info("ℹ️  Mode CPU activé")
    logger.info("   • Modèle recommandé: YOLO11n (Optimisé pour CPU)")
    logger.info("   • Conseil: GPU recommandé pour meilleures performances")
    
    return "cpu", "yolo11n.pt", False, 0


# ============================================================================
#                         CLASSES DE DONNÉES
# ============================================================================

@dataclass
class DetectionZone:
    """Zone de détection personnalisée"""
    name: str
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 255, 255)
    alert_classes: Optional[Set[str]] = None


@dataclass
class PerformanceMetrics:
    """Métriques de performance"""
    avg_fps: float = 0.0
    avg_inference_time: float = 0.0
    total_detections: int = 0
    dropped_frames: int = 0
    processing_time: float = 0.0


class ObjectDetectionConfig:
    """Configuration optimale avec auto-détection"""
    
    def __init__(self):
        # Auto-détection du matériel
        self.device, self.model_name, self.half_precision, self.gpu_memory = detect_hardware()
        
        logger.info("\n" + "=" * 80)
        logger.info("⚙️  CONFIGURATION DU SYSTÈME")
        logger.info("=" * 80)
        
        # Configuration adaptative selon le matériel
        if self.device == "cuda":
            # Configuration optimale pour GPU NVIDIA
            logger.info("🚀 Mode GPU NVIDIA - Configuration Ultra-Performance")
            self.confidence_threshold = 0.35
            self.iou_threshold = 0.5
            self.max_det = 300
            self.frame_width = 1920
            self.frame_height = 1080
            self.fps_limit = 60
            self.use_frame_skip = False
            self.trail_length = 60
            
        elif self.device == "mps":
            # Configuration pour Apple Silicon
            logger.info("🍎 Mode Apple Silicon - Configuration Optimisée")
            self.confidence_threshold = 0.4
            self.iou_threshold = 0.5
            self.max_det = 200
            self.frame_width = 1920
            self.frame_height = 1080
            self.fps_limit = 60
            self.use_frame_skip = False
            self.trail_length = 50
            
        else:
            # Configuration pour CPU
            logger.info("💻 Mode CPU - Configuration Performance/Qualité")
            self.confidence_threshold = 0.5
            self.iou_threshold = 0.5
            self.max_det = 100
            self.frame_width = 1280
            self.frame_height = 720
            self.fps_limit = 30
            self.use_frame_skip = True
            self.trail_length = 30
        
        # Paramètres communs
        self.video_source = 0
        self.save_output = True
        self.save_method = "frames"  # "video" ou "frames" (frames = plus fiable)
        self.save_with_panel = False  # False = meilleure compatibilité vidéo
        self.output_dir = "detection_results"
        self.frames_dir = None  # Sera créé si save_method="frames"
        self.show_fps = True
        self.show_confidence = True
        self.show_inference_time = True
        self.track_objects = True
        self.frame_skip_rate = 2
        
        # Features avancées
        self.enable_zones = False
        self.enable_heatmap = False
        self.min_box_area = 400
        
        # Visualisation
        self.line_thickness = 3 if self.device != "cpu" else 2
        self.font_scale = 0.7 if self.device != "cpu" else 0.5
        
        # Classes à détecter (objets du quotidien)
        self.classes_to_detect = self._get_daily_objects_classes()
        
        self._print_config()
    
    def _get_daily_objects_classes(self):
        """Retourne les classes d'objets du quotidien"""
        return [
            # Personnes
            0,   # person
            # Signalisation
            10,  # fire hydrant
            # Mobilier
            56, 57, 58, 59, 60,  # chair, couch, potted plant, bed, dining table
            # Électronique
            62, 63, 64, 65, 66, 67,  # tv, laptop, mouse, remote, keyboard, cell phone
            # Nourriture
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            # Ustensiles
            39, 40, 41, 42, 43, 44, 45,
            # Accessoires
            24, 25, 26, 27, 28,
            # Objets quotidiens
            61, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
        ]
    
    def _print_config(self):
        """Affiche la configuration détaillée"""
        logger.info(f"   • Modèle: {self.model_name}")
        logger.info(f"   • Device: {self.device.upper()}")
        logger.info(f"   • Half Precision (FP16): {'Oui' if self.half_precision else 'Non'}")
        logger.info(f"   • Résolution: {self.frame_width}x{self.frame_height}")
        logger.info(f"   • Seuil confiance: {self.confidence_threshold}")
        logger.info(f"   • Max détections: {self.max_det}")
        logger.info(f"   • Frame Skip: {'Oui' if self.use_frame_skip else 'Non'}")
        logger.info(f"   • Tracking: {'Activé' if self.track_objects else 'Désactivé'}")
        logger.info(f"   • Enregistrement: {'Activé' if self.save_output else 'Désactivé'}")
        if self.save_output:
            if self.save_method == "frames":
                logger.info(f"   • Méthode: Frame par frame (100% fiable)")
            else:
                logger.info(f"   • Méthode: Vidéo directe")
                logger.info(f"   • Panneau dans vidéo: {'Oui' if self.save_with_panel else 'Non'}")
        logger.info(f"   • Classes: {len(self.classes_to_detect)} catégories")
        logger.info("=" * 80)


# ============================================================================
#                         DÉTECTEUR D'OBJETS
# ============================================================================

class ObjectDetector:
    """Système de détection d'objets ultra-performant"""
    
    def __init__(self, config: ObjectDetectionConfig):
        logger.info("\n" + "=" * 80)
        logger.info("🚀 INITIALISATION DU SYSTÈME")
        logger.info("=" * 80)
        
        self.config = config
        self.model = None
        self.cap = None
        self.output_video = None
        
        # Structures de données
        self.track_history = defaultdict(lambda: deque(maxlen=config.trail_length))
        self.object_counts = defaultdict(int)
        self.class_statistics = defaultdict(lambda: {"count": 0, "avg_confidence": []})
        
        # Métriques
        self.fps_history = deque(maxlen=60)
        self.inference_times = deque(maxlen=60)
        self.metrics = PerformanceMetrics()
        
        # Zones et heatmap
        self.zones: List[DetectionZone] = []
        self.heatmap = None
        
        # Sauvegarde de frames
        self.saved_frames = []
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Noms des catégories
        self.category_names = {
            0: "👥 Personne",
            10: "🚒 Bouche incendie",
            56: "🪑 Chaise", 57: "🛋️  Canapé", 58: "🌿 Plante", 59: "🛏️  Lit", 60: "🍽️  Table",
            62: "📺 TV", 63: "💻 Laptop", 64: "🖱️  Souris", 65: "📱 Télécommande", 66: "⌨️  Clavier", 67: "📱 Téléphone",
            46: "🍌 Banane", 47: "🍎 Pomme", 48: "🥪 Sandwich", 49: "🍊 Orange", 50: "🥦 Brocoli",
            51: "🥕 Carotte", 52: "🌭 Hot-dog", 53: "🍕 Pizza", 54: "🍩 Donut", 55: "🎂 Gâteau",
            39: "🍾 Bouteille", 40: "🍷 Verre vin", 41: "☕ Tasse", 42: "🍴 Fourchette", 43: "🔪 Couteau",
            44: "🥄 Cuillère", 45: "🥣 Bol",
            24: "🎒 Sac à dos", 25: "☂️  Parapluie", 26: "👜 Sac", 27: "👔 Cravate", 28: "🧳 Valise",
            61: "🚽 Toilette", 68: "📟 Micro-ondes", 69: "🔥 Four", 70: "🍞 Grille-pain", 71: "🚰 Évier",
            72: "🧊 Réfrigérateur", 73: "📖 Livre", 74: "⏰ Horloge", 75: "🏺 Vase", 76: "✂️  Ciseaux",
            77: "🧸 Ours peluche", 78: "💨 Sèche-cheveux", 79: "🪥 Brosse dents"
        }
        
        # Initialisation
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self._initialize_model()
        self._initialize_video_capture()
    
    def _initialize_model(self):
        """Charge et optimise le modèle YOLO"""
        try:
            logger.info("📥 Chargement du modèle YOLO11...")
            logger.info(f"   • Fichier: {self.config.model_name}")
            
            start_time = time.time()
            self.model = YOLO(self.config.model_name)
            load_time = time.time() - start_time
            
            logger.info(f"   ✅ Modèle chargé en {load_time:.2f}s")
            
            # Optimisations
            logger.info("⚡ Optimisation du modèle...")
            
            if self.config.half_precision and self.config.device != "cpu":
                logger.info("   • Activation FP16 (half precision)...")
                self.model.to(self.config.device).half()
                logger.info("   ✅ FP16 activé - Vitesse x2")
            else:
                self.model.to(self.config.device)
                logger.info("   • Mode FP32 (précision complète)")
            
            # Warmup du modèle
            logger.info("🔥 Préchauffage du modèle (warmup)...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            
            warmup_times = []
            for i in range(3):
                start = time.time()
                _ = self.model(dummy_img, verbose=False)
                warmup_times.append(time.time() - start)
                logger.info(f"   • Warmup {i+1}/3: {warmup_times[-1]*1000:.1f}ms")
            
            avg_warmup = np.mean(warmup_times)
            logger.info(f"   ✅ Warmup terminé - Temps moyen: {avg_warmup*1000:.1f}ms")
            
            # Informations sur le modèle
            logger.info("📊 Informations du modèle:")
            logger.info(f"   • Classes disponibles: {len(self.model.names)}")
            logger.info(f"   • Classes sélectionnées: {len(self.config.classes_to_detect)}")
            logger.info(f"   • Device: {next(self.model.model.parameters()).device}")
            
            logger.info("✅ Modèle prêt et optimisé!")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
    
    def _initialize_video_capture(self):
        """Initialise la capture vidéo"""
        try:
            logger.info("\n📹 Initialisation de la capture vidéo...")
            
            self.cap = cv2.VideoCapture(self.config.video_source)
            
            if not self.cap.isOpened():
                raise ValueError(f"Impossible d'ouvrir la source vidéo: {self.config.video_source}")
            
            logger.info("   ✅ Caméra ouverte")
            
            # Configuration de la caméra
            logger.info("⚙️  Configuration de la caméra...")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps_limit)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Lecture des valeurs réelles
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"   • Résolution: {width}x{height}")
            logger.info(f"   • FPS: {fps}")
            logger.info(f"   • Backend: {self.cap.getBackendName()}")
            
            # Heatmap
            if self.config.enable_heatmap:
                self.heatmap = np.zeros((height, width), dtype=np.float32)
                logger.info("   • Heatmap activée")
            
            # Configuration de l'enregistrement
            if self.config.save_output:
                if self.config.save_method == "frames":
                    # Méthode frames (plus fiable)
                    self.config.frames_dir = Path(self.config.output_dir) / f"frames_{self.session_timestamp}"
                    self.config.frames_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"   ✅ Mode: Sauvegarde frame par frame")
                    logger.info(f"   • Dossier: {self.config.frames_dir}")
                    logger.info(f"   • Format: JPG haute qualité")
                    logger.info(f"   ⚠️  La vidéo finale sera créée à la fin")
                    
                else:
                    # Méthode vidéo directe (peut échouer)
                    self._initialize_video_writer(fps, width, height)
            
            logger.info("✅ Capture vidéo prête!")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation vidéo: {e}")
            raise
    
    def _initialize_video_writer(self, fps, width, height):
        """Initialise l'écriture vidéo directe"""
        # Essayer plusieurs codecs dans l'ordre de préférence
        codecs_to_try = [
            ('mp4v', '.mp4', 'MP4V'),
            ('XVID', '.avi', 'XVID'),
            ('MJPG', '.avi', 'MJPEG'),
            ('X264', '.mp4', 'H264'),
        ]
        
        self.output_video = None
        output_path = None
        
        for codec_str, extension, codec_name in codecs_to_try:
            try:
                output_path = Path(self.config.output_dir) / f"detection_{self.session_timestamp}{extension}"
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                
                test_writer = cv2.VideoWriter(
                    str(output_path), fourcc, fps, (width, height)
                )
                
                if test_writer.isOpened():
                    self.output_video = test_writer
                    logger.info(f"   ✅ Codec vidéo: {codec_name}")
                    logger.info(f"   ✅ Enregistrement: {output_path}")
                    logger.info(f"   • Format: {extension.upper()} à {fps} FPS")
                    return
                else:
                    test_writer.release()
                    
            except Exception as e:
                logger.warning(f"   ⚠️  Codec {codec_name} non disponible")
                continue
        
        logger.warning("   ⚠️  ATTENTION: Impossible d'initialiser l'enregistrement vidéo!")
        logger.warning("   ⚠️  Basculement vers le mode 'frames'")
        logger.warning("")
        logger.warning("   💡 SOLUTIONS POSSIBLES:")
        logger.warning("   1. Installer ffmpeg: https://ffmpeg.org/download.html")
        logger.warning("   2. Installer codecs: pip install opencv-contrib-python")
        logger.warning("   3. Le mode 'frames' sera utilisé automatiquement")
        
        # Basculer vers mode frames
        self.config.save_method = "frames"
        self.config.frames_dir = Path(self.config.output_dir) / f"frames_{self.session_timestamp}"
        self.config.frames_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Génère une couleur consistante par classe"""
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(100, 255, 3)))
    
    def _draw_boxes(self, frame: np.ndarray, results) -> np.ndarray:
        """Dessine les boîtes de détection"""
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Filtres
                if self.config.classes_to_detect and class_id not in self.config.classes_to_detect:
                    continue
                
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < self.config.min_box_area:
                    continue
                
                color = self._get_color(class_id)
                
                # Boîte avec coins marqués
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.config.line_thickness)
                
                corner_length = 15
                cv2.line(annotated_frame, (x1, y1), (x1 + corner_length, y1), color, self.config.line_thickness + 1)
                cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_length), color, self.config.line_thickness + 1)
                cv2.line(annotated_frame, (x2, y1), (x2 - corner_length, y1), color, self.config.line_thickness + 1)
                cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_length), color, self.config.line_thickness + 1)
                cv2.line(annotated_frame, (x1, y2), (x1 + corner_length, y2), color, self.config.line_thickness + 1)
                cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_length), color, self.config.line_thickness + 1)
                cv2.line(annotated_frame, (x2, y2), (x2 - corner_length, y2), color, self.config.line_thickness + 1)
                cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_length), color, self.config.line_thickness + 1)
                
                # Label avec emoji
                emoji_name = self.category_names.get(class_id, class_name)
                if self.config.show_confidence:
                    label = f"{emoji_name} {confidence:.2f}"
                else:
                    label = emoji_name
                
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 2
                )
                
                # Fond semi-transparent
                overlay = annotated_frame.copy()
                cv2.rectangle(
                    overlay,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width + 10, y1),
                    color,
                    -1
                )
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Statistiques
                self.object_counts[class_name] += 1
                self.class_statistics[class_name]["count"] += 1
                self.class_statistics[class_name]["avg_confidence"].append(confidence)
        
        return annotated_frame
    
    def _draw_tracks(self, frame: np.ndarray, results) -> np.ndarray:
        """Dessine les trajectoires de tracking"""
        if not results or not hasattr(results[0], 'boxes') or not hasattr(results[0].boxes, 'id'):
            return frame
        
        annotated_frame = frame.copy()
        
        for result in results:
            if result.boxes.id is None:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, class_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                self.track_history[track_id].append(center)
                points = list(self.track_history[track_id])
                color = self._get_color(class_id)
                
                # Trajectoire avec gradient
                if len(points) > 1:
                    for i in range(1, len(points)):
                        progress = i / len(points)
                        thickness = int(2 + progress * 3)
                        alpha = progress * 0.8
                        trail_color = tuple(int(c * alpha) for c in color)
                        cv2.line(annotated_frame, points[i-1], points[i], trail_color, thickness, cv2.LINE_AA)
                
                # ID
                id_text = f"ID:{track_id}"
                (id_w, id_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y2 + 2), (x1 + id_w + 6, y2 + id_h + 8), color, -1)
                cv2.putText(annotated_frame, id_text, (x1 + 3, y2 + id_h + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        return annotated_frame
    
    def _draw_info_panel(self, frame: np.ndarray, fps: float, inference_time: float) -> np.ndarray:
        """Panneau d'informations avancé"""
        panel_height = 200
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        
        # Gradient de fond
        for i in range(panel_height):
            alpha = i / panel_height
            panel[i, :] = (int(30 + alpha * 30), int(30 + alpha * 30), int(30 + alpha * 30))
        
        # FPS avec code couleur
        fps_color = (0, 255, 0) if fps > 30 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(panel, fps_text, (15, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, fps_color, 2, cv2.LINE_AA)
        
        # Temps d'inférence
        inf_text = f"Inference: {inference_time*1000:.1f}ms"
        cv2.putText(panel, inf_text, (220, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Modèle
        model_text = f"{self.config.model_name.upper()} | {self.config.device.upper()}"
        cv2.putText(panel, model_text, (520, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Compteurs top 5
        y_offset = 85
        sorted_counts = sorted(self.object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (class_name, count) in enumerate(sorted_counts):
            bar_width = min(count * 10, 250)
            color = self._get_color(hash(class_name) % 80)
            
            cv2.rectangle(panel, (15, y_offset + i * 22), (15 + bar_width, y_offset + i * 22 + 15), color, -1)
            
            text = f"{class_name}: {count}"
            cv2.putText(panel, text, (270, y_offset + i * 22 + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Statistiques
        total_det = sum(self.object_counts.values())
        stats_text = f"Total: {total_det} | Tracks: {len(self.track_history)} | Device: {self.config.device.upper()}"
        cv2.putText(panel, stats_text, (15, panel_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1, cv2.LINE_AA)
        
        return np.vstack([frame, panel])
    
    def detect(self):
        """Boucle principale de détection avec logs détaillés"""
        logger.info("\n" + "=" * 80)
        logger.info("🎬 DÉMARRAGE DE LA DÉTECTION")
        logger.info("=" * 80)
        logger.info("📝 CONTRÔLES:")
        logger.info("   • Appuyez sur 'Q' pour arrêter")
        logger.info("   • Appuyez sur 'ESC' pour arrêter")
        logger.info("   • La fenêtre OpenCV doit être ACTIVE (cliquée)")
        logger.info("=" * 80 + "\n")
        
        # S'assurer que la fenêtre est active
        cv2.namedWindow('YOLO11 Ultimate Detection [Q=Quit]', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        skip_counter = 0
        start_session = time.time()
        last_key_check = time.time()
        
        try:
            while True:
                loop_start = time.time()
                
                # Lecture frame
                ret, frame = self.cap.read()
                if not ret:
                    self.metrics.dropped_frames += 1
                    if self.metrics.dropped_frames > 10:
                        logger.warning("⚠️  Trop d'images perdues. Arrêt...")
                        break
                    continue
                
                # Frame skipping
                if self.config.use_frame_skip:
                    skip_counter += 1
                    if skip_counter % self.config.frame_skip_rate != 0:
                        continue
                
                self.object_counts.clear()
                
                # Inférence
                inference_start = time.time()
                
                if self.config.track_objects:
                    results = self.model.track(
                        frame,
                        conf=self.config.confidence_threshold,
                        iou=self.config.iou_threshold,
                        max_det=self.config.max_det,
                        persist=True,
                        classes=self.config.classes_to_detect,
                        device=self.config.device,
                        verbose=False,
                        half=self.config.half_precision
                    )
                else:
                    results = self.model(
                        frame,
                        conf=self.config.confidence_threshold,
                        iou=self.config.iou_threshold,
                        max_det=self.config.max_det,
                        classes=self.config.classes_to_detect,
                        device=self.config.device,
                        verbose=False,
                        half=self.config.half_precision
                    )
                
                inference_time = time.time() - inference_start
                self.inference_times.append(inference_time)
                
                # Annotations
                annotated_frame = self._draw_boxes(frame, results)
                
                if self.config.track_objects:
                    annotated_frame = self._draw_tracks(annotated_frame, results)
                
                # Métriques
                fps = 1.0 / (time.time() - loop_start)
                self.fps_history.append(fps)
                avg_fps = np.mean(self.fps_history)
                avg_inference = np.mean(self.inference_times)
                
                self.metrics.avg_fps = avg_fps
                self.metrics.avg_inference_time = avg_inference
                self.metrics.total_detections += sum(self.object_counts.values())
                
                # Panneau info
                final_frame = self._draw_info_panel(annotated_frame, avg_fps, avg_inference)
                
                # Affichage
                window_name = 'YOLO11 Ultimate Detection [Q=Quit]'
                cv2.imshow(window_name, final_frame)
                
                # IMPORTANT: Forcer le rafraîchissement de la fenêtre
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                
                # Sauvegarde
                if self.config.save_output:
                    frame_to_save = final_frame if self.config.save_with_panel else annotated_frame
                    
                    if self.config.save_method == "frames":
                        # Sauvegarde frame par frame (méthode la plus fiable)
                        frame_filename = self.config.frames_dir / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_filename), frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        self.saved_frames.append(frame_filename)
                        
                    elif self.output_video is not None and self.output_video.isOpened():
                        # Sauvegarde vidéo directe
                        target_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        target_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        if frame_to_save.shape[1] == target_width and frame_to_save.shape[0] == target_height:
                            self.output_video.write(frame_to_save)
                        else:
                            resized_frame = cv2.resize(frame_to_save, (target_width, target_height))
                            self.output_video.write(resized_frame)
                
                frame_count += 1
                
                # Logs périodiques
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_session
                    logger.info(f"📊 Frame {frame_count:5d} | FPS: {avg_fps:5.1f} | "
                              f"Inference: {avg_inference*1000:5.1f}ms | "
                              f"Détections: {self.metrics.total_detections:6d} | "
                              f"Temps: {elapsed/60:.1f}min")
                
                # Sortie
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    logger.info("\n⏹️  Arrêt demandé par l'utilisateur")
                    break
                
        except KeyboardInterrupt:
            logger.info("\n⏹️  Interruption clavier (Ctrl+C)")
        except Exception as e:
            logger.error(f"\n❌ Erreur pendant la détection: {e}", exc_info=True)
        finally:
            logger.info("\n" + "=" * 80)
            logger.info(f"📊 RÉSUMÉ: {frame_count} frames traitées en {(time.time()-start_session)/60:.1f} min")
            logger.info("=" * 80)
            self._cleanup()
            self._save_statistics(frame_count, time.time() - start_session)
    
    def _cleanup(self):
        """Nettoyage des ressources"""
        logger.info("\n" + "=" * 80)
        logger.info("🧹 NETTOYAGE DES RESSOURCES")
        logger.info("=" * 80)
        
        if self.cap:
            self.cap.release()
            logger.info("   ✅ Caméra fermée")
        
        # Gestion de la sauvegarde selon la méthode
        if self.config.save_output:
            if self.config.save_method == "frames" and self.saved_frames:
                logger.info(f"   ✅ {len(self.saved_frames)} frames sauvegardées")
                logger.info("   ⏳ Création de la vidéo finale...")
                self._create_video_from_frames()
                
            elif self.output_video:
                try:
                    logger.info("   ⏳ Finalisation de la vidéo...")
                    self.output_video.release()
                    self._verify_video_file()
                except Exception as e:
                    logger.error(f"   ❌ Erreur lors de la fermeture de la vidéo: {e}")
        
        cv2.destroyAllWindows()
        logger.info("   ✅ Fenêtres fermées")
    
    def _create_video_from_frames(self):
        """Crée une vidéo à partir des frames sauvegardées"""
        try:
            if not self.saved_frames:
                logger.warning("   ⚠️  Aucune frame à compiler")
                return
            
            logger.info(f"   📁 {len(self.saved_frames)} frames à compiler")
            
            # Lire la première frame pour obtenir les dimensions
            first_frame = cv2.imread(str(self.saved_frames[0]))
            if first_frame is None:
                logger.error("   ❌ Impossible de lire la première frame")
                logger.error(f"   • Fichier: {self.saved_frames[0]}")
                return
            
            height, width = first_frame.shape[:2]
            logger.info(f"   • Dimensions détectées: {width}x{height}")
            
            # FPS de la capture (avec fallback)
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 30
            if fps <= 0 or fps > 120:
                fps = 30
                logger.warning(f"   ⚠️  FPS invalide, utilisation de {fps} FPS par défaut")
            logger.info(f"   • FPS: {fps}")
            
            # Durée estimée
            duration = len(self.saved_frames) / fps
            logger.info(f"   • Durée estimée: {duration:.1f} secondes ({duration/60:.1f} minutes)")
            
            # Créer la vidéo avec différents codecs
            output_path = Path(self.config.output_dir) / f"detection_{self.session_timestamp}_final.mp4"
            
            # Essayer plusieurs codecs
            codecs = [
                (cv2.VideoWriter_fourcc(*'mp4v'), '.mp4', 'MP4V'),
                (cv2.VideoWriter_fourcc(*'X264'), '.mp4', 'X264'),
                (cv2.VideoWriter_fourcc(*'MJPG'), '.avi', 'MJPEG'),
                (cv2.VideoWriter_fourcc(*'XVID'), '.avi', 'XVID'),
            ]
            
            video_writer = None
            for fourcc, ext, codec_name in codecs:
                try:
                    test_path = output_path.with_suffix(ext)
                    logger.info(f"   • Tentative avec codec {codec_name}...")
                    
                    video_writer = cv2.VideoWriter(
                        str(test_path), 
                        fourcc, 
                        fps, 
                        (width, height),
                        True  # isColor
                    )
                    
                    if video_writer is not None and video_writer.isOpened():
                        output_path = test_path
                        logger.info(f"   ✅ Codec sélectionné: {codec_name} ({ext})")
                        break
                    else:
                        if video_writer:
                            video_writer.release()
                        video_writer = None
                        logger.warning(f"   ⚠️  {codec_name} non disponible")
                except Exception as e:
                    logger.warning(f"   ⚠️  Erreur avec {codec_name}: {e}")
                    if video_writer:
                        video_writer.release()
                    video_writer = None
                    continue
            
            if video_writer is None or not video_writer.isOpened():
                logger.error("   ❌ Impossible de créer la vidéo avec aucun codec")
                logger.info("   💡 Les frames sont disponibles dans:")
                logger.info(f"      {self.config.frames_dir}")
                logger.info("")
                logger.info("   💡 SOLUTION MANUELLE:")
                logger.info("   1. Installez FFmpeg: https://ffmpeg.org/download.html")
                logger.info("   2. Exécutez cette commande:")
                logger.info(f"      cd {self.config.frames_dir}")
                logger.info(f"      ffmpeg -framerate {fps} -i frame_%06d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4")
                return
            
            # Écrire toutes les frames avec barre de progression
            logger.info(f"   ⏳ Compilation en cours...")
            logger.info("")
            
            frames_written = 0
            frames_failed = 0
            
            for i, frame_path in enumerate(self.saved_frames):
                try:
                    frame = cv2.imread(str(frame_path))
                    
                    if frame is None:
                        frames_failed += 1
                        logger.warning(f"   ⚠️  Frame {i+1} corrompue, ignorée")
                        continue
                    
                    # Vérifier et redimensionner si nécessaire
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    
                    # Écrire la frame
                    video_writer.write(frame)
                    frames_written += 1
                    
                    # Afficher la progression tous les 10%
                    progress = (i + 1) / len(self.saved_frames)
                    if (i + 1) % max(1, len(self.saved_frames) // 10) == 0:
                        logger.info(f"   • Progression: {progress*100:.0f}% ({i+1}/{len(self.saved_frames)})")
                
                except Exception as e:
                    frames_failed += 1
                    logger.warning(f"   ⚠️  Erreur frame {i+1}: {e}")
                    continue
            
            # Fermer le writer
            video_writer.release()
            logger.info("")
            logger.info(f"   ✅ Écriture terminée: {frames_written} frames écrites, {frames_failed} échecs")
            
            # Attendre un peu pour s'assurer que le fichier est bien fermé
            import time
            time.sleep(0.5)
            
            # Vérifier le fichier créé
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)
                
                if file_size < 0.1:
                    logger.error(f"   ❌ Vidéo trop petite ({file_size*1024:.1f} KB)")
                    logger.error("   • Le fichier semble corrompu ou vide")
                else:
                    logger.info(f"   ✅ Vidéo créée avec succès!")
                    logger.info(f"   • Fichier: {output_path.name}")
                    logger.info(f"   • Taille: {file_size:.1f} MB")
                    logger.info(f"   • Frames écrites: {frames_written}")
                    logger.info(f"   • Résolution: {width}x{height}")
                    logger.info(f"   • FPS: {fps}")
                    logger.info(f"   • Durée théorique: {frames_written/fps:.1f}s")
                    logger.info(f"   • Chemin: {output_path.absolute()}")
                    
                    # Tester si lisible
                    logger.info("")
                    logger.info("   🔍 Vérification de la vidéo...")
                    try:
                        test_cap = cv2.VideoCapture(str(output_path))
                        if test_cap.isOpened():
                            frame_count_check = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            duration_check = frame_count_check / test_cap.get(cv2.CAP_PROP_FPS)
                            width_check = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height_check = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps_check = test_cap.get(cv2.CAP_PROP_FPS)
                            test_cap.release()
                            
                            logger.info(f"   ✅ VIDÉO VÉRIFIÉE:")
                            logger.info(f"   • Frames lisibles: {frame_count_check}")
                            logger.info(f"   • Durée: {duration_check:.1f}s ({duration_check/60:.1f} min)")
                            logger.info(f"   • Résolution: {width_check}x{height_check}")
                            logger.info(f"   • FPS: {fps_check:.1f}")
                            
                            if frame_count_check > 0 and duration_check > 0:
                                logger.info("")
                                logger.info("   ✅ ✅ ✅ LA VIDÉO EST PARFAITEMENT LISIBLE! ✅ ✅ ✅")
                                logger.info("")
                                logger.info("   📹 Pour lire la vidéo:")
                                logger.info("      • VLC Media Player (recommandé)")
                                logger.info("      • Windows Media Player")
                                logger.info("      • QuickTime (Mac)")
                                logger.info("      • mpv (Linux)")
                            else:
                                logger.warning("   ⚠️  La vidéo a 0 frame ou 0 durée")
                                logger.warning("   • OpenCV a peut-être mal lu les métadonnées")
                                logger.warning("   • Essayez quand même de l'ouvrir avec VLC")
                        else:
                            logger.warning("   ⚠️  OpenCV ne peut pas ouvrir la vidéo")
                            logger.warning("   • Cela ne signifie pas qu'elle est corrompue")
                            logger.warning("   • Essayez de l'ouvrir avec VLC Media Player")
                    except Exception as e:
                        logger.warning(f"   ⚠️  Impossible de vérifier la vidéo: {e}")
                        logger.warning("   • Essayez de l'ouvrir avec VLC Media Player")
                    
                    logger.info("")
                    logger.info("   💡 Les frames originales sont dans:")
                    logger.info(f"      {self.config.frames_dir}")
                    logger.info("   💡 Vous pouvez les supprimer pour économiser de l'espace")
            else:
                logger.error("   ❌ Le fichier vidéo n'a pas été créé")
                logger.info("   💡 Vérifiez les permissions du dossier")
                
        except Exception as e:
            logger.error(f"   ❌ Erreur lors de la création de la vidéo: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("")
            logger.info("   💡 Les frames individuelles sont disponibles dans:")
            logger.info(f"      {self.config.frames_dir}")
            logger.info("")
            logger.info("   💡 SOLUTION ALTERNATIVE avec FFmpeg:")
            logger.info("   1. Téléchargez FFmpeg: https://ffmpeg.org/download.html")
            logger.info("   2. Exécutez:")
            logger.info(f"      cd {self.config.frames_dir}")
            logger.info(f"      ffmpeg -framerate {fps} -i frame_%06d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4")
    
    def _verify_video_file(self):
        """Vérifie le fichier vidéo créé en mode direct"""
        try:
            video_files = list(Path(self.config.output_dir).glob("detection_*.mp4"))
            video_files.extend(list(Path(self.config.output_dir).glob("detection_*.avi")))
            
            if video_files:
                latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
                file_size = latest_video.stat().st_size / (1024 * 1024)
                
                if file_size > 0:
                    logger.info(f"   ✅ Vidéo sauvegardée: {latest_video.name}")
                    logger.info(f"   • Taille: {file_size:.1f} MB")
                    logger.info(f"   • Chemin: {latest_video.absolute()}")
                    
                    # Vérifier si lisible
                    test_cap = cv2.VideoCapture(str(latest_video))
                    if test_cap.isOpened():
                        frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        test_cap.release()
                        
                        if frame_count > 0:
                            logger.info(f"   ✅ VIDÉO VÉRIFIÉE ({frame_count} frames)")
                        else:
                            logger.warning("   ⚠️  La vidéo semble vide")
                    else:
                        logger.warning("   ⚠️  Impossible d'ouvrir la vidéo")
                else:
                    logger.warning(f"   ⚠️  Fichier vidéo vide")
            else:
                logger.warning("   ⚠️  Aucun fichier vidéo trouvé")
                
        except Exception as e:
            logger.warning(f"   ⚠️  Erreur vérification: {e}")
    
    def _save_statistics(self, frame_count: int, session_duration: float):
        """Sauvegarde des statistiques détaillées"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 GÉNÉRATION DES STATISTIQUES")
        logger.info("=" * 80)
        
        # Statistiques par classe
        class_stats = {}
        for class_name, data in self.class_statistics.items():
            if data["avg_confidence"]:
                class_stats[class_name] = {
                    "total_count": data["count"],
                    "avg_confidence": float(np.mean(data["avg_confidence"])),
                    "min_confidence": float(np.min(data["avg_confidence"])),
                    "max_confidence": float(np.max(data["avg_confidence"]))
                }
        
        stats = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": session_duration,
                "total_frames": frame_count,
                "dropped_frames": self.metrics.dropped_frames
            },
            "performance": {
                "average_fps": float(self.metrics.avg_fps),
                "average_inference_time_ms": float(self.metrics.avg_inference_time * 1000),
                "min_fps": float(np.min(self.fps_history)) if self.fps_history else 0,
                "max_fps": float(np.max(self.fps_history)) if self.fps_history else 0,
                "min_inference_ms": float(np.min(self.inference_times) * 1000) if self.inference_times else 0,
                "max_inference_ms": float(np.max(self.inference_times) * 1000) if self.inference_times else 0
            },
            "model_config": {
                "model": self.config.model_name,
                "device": self.config.device,
                "half_precision": self.config.half_precision,
                "confidence_threshold": self.config.confidence_threshold,
                "resolution": f"{self.config.frame_width}x{self.config.frame_height}"
            },
            "detection_stats": {
                "total_detections": self.metrics.total_detections,
                "unique_tracks": len(self.track_history),
                "class_statistics": class_stats
            }
        }
        
        # Sauvegarde JSON
        stats_path = Path(self.config.output_dir) / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        
        logger.info(f"   ✅ Statistiques: {stats_path}")
        
        # Rapport final
        logger.info("\n" + "=" * 80)
        logger.info("📈 RAPPORT FINAL DE SESSION")
        logger.info("=" * 80)
        logger.info(f"⏱️  Durée: {session_duration/60:.1f} minutes")
        logger.info(f"🎞️  Frames: {frame_count}")
        logger.info(f"📊 FPS moyen: {self.metrics.avg_fps:.1f} (min: {stats['performance']['min_fps']:.1f}, max: {stats['performance']['max_fps']:.1f})")
        logger.info(f"⚡ Inférence: {self.metrics.avg_inference_time*1000:.1f}ms (min: {stats['performance']['min_inference_ms']:.1f}ms, max: {stats['performance']['max_inference_ms']:.1f}ms)")
        logger.info(f"🎯 Détections: {self.metrics.total_detections}")
        logger.info(f"🔢 Tracks uniques: {len(self.track_history)}")
        logger.info(f"📦 Classes détectées: {len(class_stats)}")
        
        if class_stats:
            logger.info("\n🏆 TOP 5 DES OBJETS DÉTECTÉS:")
            sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['total_count'], reverse=True)[:5]
            for i, (name, data) in enumerate(sorted_classes, 1):
                logger.info(f"   {i}. {name}: {data['total_count']} détections (confiance moy: {data['avg_confidence']:.2f})")
        
        logger.info("=" * 80)
        logger.info("✅ SESSION TERMINÉE AVEC SUCCÈS")
        logger.info("=" * 80 + "\n")


# ============================================================================
#                         FONCTION PRINCIPALE
# ============================================================================

def print_banner():
    """Affiche la bannière du programme"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               ██    ██  ██████  ██       ██████   ██ ██                      ║
║                ██  ██  ██    ██ ██      ██    ██ ███ ███                     ║
║                 ████   ██    ██ ██      ██    ██  ██  ██                     ║
║                  ██    ██    ██ ██      ██    ██  ██  ██                     ║
║                  ██     ██████  ███████  ██████   ██  ██                     ║
║                                                                              ║
║                                                                              ║
║                                                                              ║
║   Fonctionnalités:                                                           ║
║   • Auto-détection matériel (GPU/CPU)                                        ║
║   • Modèle adaptatif selon performances                                      ║
║   • Tracking temps réel avec trajectoires                                    ║
║   • Statistiques détaillées                                                  ║
║   • Support CUDA / Apple Silicon / CPU                                       ║
║   • Logs explicatifs en français                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Point d'entrée principal avec logs détaillés"""
    print_banner()
    
    try:
        # Configuration automatique
        logger.info("🚀 Démarrage du système YOLO11 Ultimate...")
        config = ObjectDetectionConfig()
        
        # Création du détecteur
        detector = ObjectDetector(config)
        
        # Démarrage de la détection
        detector.detect()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"\n❌ ERREUR CRITIQUE: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()