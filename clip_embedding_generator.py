# clip_embedding_generator_png_optimized.py
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import re
import io

# CLIP imports with GPU verification
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
    print("✅ CLIP is available")
    # GPU verification
    if torch.cuda.is_available():
        print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    else:
        print("⚠️ CUDA not available, using CPU")
except ImportError:
    print("❌ CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False

# Visual processing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("Installing OpenCV...")
    import subprocess
    subprocess.check_call(["pip", "install", "opencv-python"])
    import cv2
    OPENCV_AVAILABLE = True

logger = logging.getLogger(__name__)

class FinnishFloorPlanCLIPProcessor:
    """GPU-optimized CLIP processor for Finnish floor plans using PNG images"""
    
    def __init__(self, rag_store_path: str):
        self.rag_store_path = Path(rag_store_path)
        
        # GPU setup
        if torch.cuda.is_available():
            self.device = 'cuda'
            torch.cuda.empty_cache()
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("⚠️ GPU not available, using CPU")
        
        # Initialize CLIP with GPU optimization
        if CLIP_AVAILABLE:
            print("🔄 Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()  # Set to eval mode for inference
            print(f"✅ CLIP model loaded on {self.device}")
        else:
            raise ImportError("CLIP not available")
        
        # Comprehensive Finnish room mapping
        self.finnish_room_mapping = {
            # From vector_store.py finnish_room_types
            'oh': {
                'full_name': 'olohuone', 'english': 'living_room', 'function': 'social_relaxation',
                'privacy': 'semi-private', 'typical_size': 'large', 'type': 'social'
            },
            'mh': {
                'full_name': 'makuuhuone', 'english': 'bedroom', 'function': 'rest_sleep',
                'privacy': 'private', 'typical_size': 'medium', 'type': 'private'
            },
            'kh': {
                'full_name': 'kylpyhuone', 'english': 'bathroom', 'function': 'hygiene_bathing',
                'privacy': 'private', 'typical_size': 'small', 'type': 'wet'
            },
            'vh': {
                'full_name': 'vesihuone', 'english': 'utility_room', 'function': 'utility_cleaning',
                'privacy': 'service', 'typical_size': 'small', 'type': 'service'
            },
            'wc': {
                'full_name': 'wc', 'english': 'toilet', 'function': 'sanitation',
                'privacy': 'private', 'typical_size': 'small', 'type': 'wet'
            },
            'keittiö': {
                'full_name': 'keittiö', 'english': 'kitchen', 'function': 'food_preparation',
                'privacy': 'semi-private', 'typical_size': 'medium', 'type': 'service'
            },
            
            # Extended Finnish terms
            'k': {'english': 'kitchen', 'finnish': 'keittiö', 'type': 'service'},
            'kk': {'english': 'kitchenette', 'finnish': 'keittokomero', 'type': 'service'},
            's': {'english': 'sauna', 'finnish': 'sauna', 'type': 'wellness'},
            'sauna': {'english': 'sauna', 'finnish': 'sauna', 'type': 'wellness'},
            'et': {'english': 'hallway', 'finnish': 'eteinen', 'type': 'circulation'},
            'eteinen': {'english': 'hallway', 'finnish': 'eteinen', 'type': 'circulation'},
            'var': {'english': 'storage', 'finnish': 'varasto', 'type': 'storage'},
            'varasto': {'english': 'storage', 'finnish': 'varasto', 'type': 'storage'},
            'tekn': {'english': 'technical_room', 'finnish': 'tekninen_tila', 'type': 'technical'},
            'pvk': {'english': 'balcony', 'finnish': 'parveke', 'type': 'outdoor'},
            'p': {'english': 'balcony', 'finnish': 'parveke', 'type': 'outdoor'},
            'parveke': {'english': 'balcony', 'finnish': 'parveke', 'type': 'outdoor'},
            'ter': {'english': 'terrace', 'finnish': 'terassi', 'type': 'outdoor'},
            'terassi': {'english': 'terrace', 'finnish': 'terassi', 'type': 'outdoor'},
            
            # From your metadata (cl, cb, cwh, sink, etc.)
            'cl': {'english': 'closet', 'finnish': 'kaappi', 'type': 'storage'},
            'cb': {'english': 'cabinet', 'finnish': 'kaappi', 'type': 'storage'},
            'cwh': {'english': 'clothes_room', 'finnish': 'vaatehuone', 'type': 'storage'},
            'sink': {'english': 'sink', 'finnish': 'pesuallas', 'type': 'fixture'},
            'undefined': {'english': 'undefined', 'finnish': 'määrittelemätön', 'type': 'unknown'},
            
            # Additional comprehensive Finnish terms
            'rh': {'english': 'dining_room', 'finnish': 'ruokailuhuone', 'type': 'social'},
            'työh': {'english': 'office', 'finnish': 'työhuone', 'type': 'work'},
            'ph': {'english': 'washing_room', 'finnish': 'pesuhuone', 'type': 'wet'},
            'sh': {'english': 'shower_room', 'finnish': 'suihkuhuone', 'type': 'wet'},
            'pkh': {'english': 'dressing_room', 'finnish': 'pukuhuone', 'type': 'storage'},
            'sk': {'english': 'cleaning_closet', 'finnish': 'siivouskomero', 'type': 'service'},
            'at': {'english': 'garage', 'finnish': 'autotalli', 'type': 'storage'},
            'ak': {'english': 'carport', 'finnish': 'autokatos', 'type': 'outdoor'},
        }
        
        # Load existing metadata
        self.metadata = self._load_existing_metadata()
        self.plan_mapping = self._load_plan_mapping()
        
        print(f"📊 Loaded metadata for {len(self.metadata)} annotations")
        print(f"🏠 Supporting {len(self.finnish_room_mapping)} Finnish room types")

    def load_floorplan_image_optimized(self, floorplan_dir: Path, size: Tuple[int, int] = (800, 600)) -> Optional[np.ndarray]:
        """
        Load PNG image from the floorplan directory (much faster than SVG conversion)
        Priority: F1_original.png > F1_scaled.png > F2_original.png > F2_scaled.png > SVG fallback
        """
        # PNG candidates in priority order
        png_candidates = [
            "F1_original.png",
            "F1_scaled.png", 
            "F2_original.png",
            "F2_scaled.png"
        ]
        
        # Try loading PNG files first
        for png_file in png_candidates:
            png_path = floorplan_dir / png_file
            if png_path.exists():
                try:
                    print(f"📸 Loading {png_file}...")
                    img = Image.open(png_path).convert('RGB')
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    return np.array(img)
                except Exception as e:
                    print(f"❌ Failed to load {png_file}: {e}")
                    continue
        
        # Fallback to SVG if no PNG found
        svg_path = floorplan_dir / "model.svg"
        if svg_path.exists():
            print(f"⚠️ No PNG found, attempting SVG conversion...")
            return self.svg_to_image_enhanced(svg_path=svg_path, size=size)
        
        # Last resort: create template
        print(f"⚠️ No images found, creating template...")
        return self._create_template_from_metadata(size)

    def svg_to_image_enhanced(self, svg_path: Path, size: Tuple[int, int] = (800, 600)) -> Optional[np.ndarray]:
        """Fallback SVG to image conversion"""
        try:
            # Try cairosvg with file reading instead of URL
            try:
                import cairosvg
                # Read SVG content first
                with open(svg_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                png_data = cairosvg.svg2png(bytestring=svg_content, output_width=size[0], output_height=size[1])
                img = Image.open(io.BytesIO(png_data)).convert('RGB')
                return np.array(img)
            except ImportError:
                print("⚠️ cairosvg not available, creating approximate image...")
                return self._create_approximate_floor_plan_image(svg_path, size)
        except Exception as e:
            print(f"❌ SVG conversion failed: {e}")
            return self._create_fallback_image(size)

    def process_png_dataset(self, dataset_dirs: List[Path], output_path: Path) -> np.ndarray:
        """Process dataset directories containing PNG images"""
        print(f"🎨 Processing {len(dataset_dirs)} floor plan directories with PNG images...")
        
        # GPU memory management
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print(f"💾 Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        all_visual_embeddings = []
        processed_files = []
        failed_files = []
        
        for i, plan_dir in enumerate(dataset_dirs, 1):
            try:
                if i % 50 == 0 or i == len(dataset_dirs):
                    gpu_info = f" (GPU mem: {torch.cuda.memory_allocated() / 1024**2:.1f} MB)" if self.device == 'cuda' else ""
                    print(f"[{i}/{len(dataset_dirs)}] Processing {plan_dir.name}...{gpu_info}")
                
                # Load PNG image (much faster than SVG)
                floor_plan_image = self.load_floorplan_image_optimized(plan_dir)
                
                if floor_plan_image is None:
                    print(f"❌ Failed to load image from {plan_dir.name}")
                    failed_files.append(plan_dir.name)
                    continue
                
                # GPU-optimized CLIP extraction
                clip_embedding = self.extract_clip_features_from_image(floor_plan_image)
                
                if clip_embedding is None:
                    print(f"❌ Failed to extract CLIP embedding for {plan_dir.name}")
                    failed_files.append(plan_dir.name)
                    continue
                
                # Create embeddings for annotations (limit per file to avoid memory issues)
                batch_size = min(1000, len(self.metadata))
                for j in range(0, batch_size):
                    if j < len(self.metadata):
                        annotation_data = self.metadata[j]
                        enhanced_embedding = self._enhance_clip_with_local_features(
                            clip_embedding, annotation_data, floor_plan_image
                        )
                        all_visual_embeddings.append(enhanced_embedding)
                
                processed_files.append(plan_dir.name)
                
                # GPU memory cleanup every 20 files
                if self.device == 'cuda' and i % 20 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ Error processing {plan_dir.name}: {e}")
                failed_files.append(plan_dir.name)
                continue
        
        if not all_visual_embeddings:
            raise ValueError("No visual embeddings were created")
        
        # Convert to numpy array
        visual_embeddings_array = np.vstack(all_visual_embeddings)
        
        # Save visual embeddings
        output_path.mkdir(parents=True, exist_ok=True)
        visual_embeddings_file = output_path / "visual_embeddings.npy"
        np.save(visual_embeddings_file, visual_embeddings_array)
        
        # Save processing log
        log_file = output_path / "visual_processing_log.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Successfully processed directories ({len(processed_files)}):\n")
            for file in processed_files:
                f.write(f"  {file}\n")
            f.write(f"\nFailed directories ({len(failed_files)}):\n")
            for file in failed_files:
                f.write(f"  {file}\n")
        
        gpu_peak = f"GPU Memory Peak: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB" if self.device == 'cuda' else "CPU Processing"
        
        print(f"✅ Saved {len(all_visual_embeddings)} visual embeddings to {visual_embeddings_file}")
        print(f"   Shape: {visual_embeddings_array.shape}")
        print(f"   Size: {visual_embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"   Processed: {len(processed_files)} directories")
        print(f"   Failed: {len(failed_files)} directories")
        print(f"   {gpu_peak}")
        
        return visual_embeddings_array

    def _load_existing_metadata(self) -> List[Dict]:
        """Load existing consolidated metadata"""
        metadata_path = self.rag_store_path / "consolidated_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    def _load_plan_mapping(self) -> Dict:
        """Load existing plan mapping"""
        mapping_path = self.rag_store_path / "plan_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                return json.load(f)
        return {}
    
    def extract_clip_features_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """GPU-optimized CLIP feature extraction"""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # GPU-optimized processing
            with torch.no_grad():  # Disable gradients for inference
                image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device, non_blocking=True)
                image_features = self.clip_model.encode_image(image_input)
                embedding = image_features.cpu().numpy().flatten()
                return embedding / np.linalg.norm(embedding)
                
        except Exception as e:
            logger.warning(f"GPU CLIP extraction failed: {e}")
            return None
    
    def _enhance_clip_with_local_features(self, base_clip_embedding: np.ndarray, 
                                        annotation_data: Dict, 
                                        floor_plan_image: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced with comprehensive Finnish room understanding"""
        try:
            # Start with base CLIP embedding
            enhanced_embedding = base_clip_embedding.copy()
            
            # Get room info from comprehensive Finnish mapping
            room_type = annotation_data.get('room_type', '').lower()
            room_info = self.finnish_room_mapping.get(room_type, {})
            
            # Enhanced room type weighting based on Finnish architecture
            finnish_room_weights = {
                'private': 0.15,    # bedrooms, offices - important in Finnish homes
                'social': 0.20,     # living rooms, dining rooms - central spaces
                'wet': 0.08,        # bathrooms, saunas - specialized
                'service': 0.12,    # kitchens, utility rooms - functional
                'storage': 0.05,    # closets, storage rooms - less prominent
                'circulation': 0.03, # hallways, entrances - transitional
                'outdoor': 0.10,    # balconies, terraces - important in Finland
                'technical': 0.02,  # technical rooms - hidden
                'wellness': 0.12,   # sauna - very Finnish!
                'work': 0.13,       # offices, studies - modern need
                'fixture': 0.02,    # sinks, built-ins - elements
                'unknown': 0.03     # unclassified
            }
            
            # Apply room type weighting
            room_category = room_info.get('type', 'unknown')
            weight = finnish_room_weights.get(room_category, 0.05)
            enhanced_embedding = enhanced_embedding * (1 + weight)
            
            # Add Finnish-specific spatial context
            spatial_features = np.array([
                annotation_data.get('global_idx', 0) / len(self.metadata),  # Position in sequence
                1.0 if room_category in ['wellness', 'outdoor'] else 0.0,   # Finnish-specific features
                1.0 if 'sauna' in annotation_data.get('text', '').lower() else 0.0,  # Sauna detection
                1.0 if room_category == 'wet' else 0.0,  # Wet room indicator
                1.0 if annotation_data.get('has_visual_features', False) else 0.0,
                1.0 if room_type in ['cl', 'cb', 'cwh'] else 0.0,  # Storage indicators from your metadata
                1.0 if room_type == 'sink' else 0.0,  # Fixture indicator
                1.0 if room_type == 'undefined' else 0.0,  # Undefined spaces
            ])
            
            # Extend embedding with enhanced spatial features
            if len(spatial_features) > 0:
                # Pad spatial features to match embedding size
                spatial_padded = np.pad(spatial_features, (0, max(0, 16 - len(spatial_features))))[:16]
                enhanced_embedding = np.concatenate([enhanced_embedding, spatial_padded])
            
            # Normalize final embedding
            return enhanced_embedding / np.linalg.norm(enhanced_embedding)
            
        except Exception as e:
            logger.warning(f"Failed to enhance CLIP embedding: {e}")
            return base_clip_embedding
    
    def _create_template_from_metadata(self, size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """Create a template floor plan image based on metadata"""
        print("🏗️ Creating template floor plan from metadata...")
        
        # Create basic Finnish floor plan layout
        img = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw typical Finnish apartment layout
        # Outer walls
        draw.rectangle([20, 20, size[0]-20, size[1]-20], outline='black', width=4)
        
        # Typical room divisions
        draw.line([size//3, 20, size//3, size[1]-20], fill='black', width=2)  # Kitchen separator
        draw.line([20, size[1]//2, size-20, size[1]//2], fill='black', width=2)  # Living/bedroom separator
        
        # Mark room areas with Finnish labels
        room_areas = [
            (size//4, size[1]//4, 'OH'),   # Living room
            (3*size//4, size[1]//4, 'MH'),   # Bedroom
            (3*size//4, 3*size[1]//4, 'KH'),   # Bathroom
            (size//4, size[1]//8, 'keittiö'), # Kitchen
        ]
        
        for x, y, label in room_areas:
            draw.circle([x, y], 8, fill='red')
            
        return np.array(img)

    def _create_approximate_floor_plan_image(self, svg_path: Path, size: Tuple[int, int]) -> np.ndarray:
        """Create approximate floor plan image"""
        return self._create_template_from_metadata(size)

    def _create_fallback_image(self, size: Tuple[int, int]) -> np.ndarray:
        """Create fallback white image"""
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255


def main():
    """Main function to generate CLIP visual embeddings from PNG dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CLIP visual embeddings for Finnish floor plans using PNG images")
    
    # Required arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the dataset containing floor plan directories")
    parser.add_argument("--rag_store_path", type=str, default="enhanced_multimodal_rag_store",
                       help="Path to your existing RAG store")
    
    # Optional arguments
    parser.add_argument("--output_path", type=str, default=None,
                       help="Custom output path (defaults to rag_store_path)")
    parser.add_argument("--max_dirs", type=int, default=None,
                       help="Maximum number of directories to process")
    
    args = parser.parse_args()
    
    print("🎨 CLIP Visual Embeddings Generator for Finnish Floor Plans (PNG Optimized)")
    print("=" * 70)
    print(f"📁 Dataset path: {args.dataset_path}")
    print(f"📂 RAG store path: {args.rag_store_path}")
    
    # Validate paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    rag_store_path = Path(args.rag_store_path)
    if not rag_store_path.exists():
        print(f"❌ RAG store path not found: {rag_store_path}")
        return
    
    if not CLIP_AVAILABLE:
        print("❌ CLIP not available. Install with:")
        print("   pip install git+https://github.com/openai/CLIP.git")
        return
    
    try:
        # Find directories containing floor plans
        plan_dirs = find_floorplan_directories(dataset_path)
        if args.max_dirs:
            plan_dirs = plan_dirs[:args.max_dirs]
        
        print(f"📊 Found {len(plan_dirs)} floor plan directories to process")
        
        # Initialize processor
        processor = FinnishFloorPlanCLIPProcessor(args.rag_store_path)
        
        # Process PNG files and generate embeddings
        visual_embeddings = processor.process_png_dataset(
            dataset_dirs=plan_dirs,
            output_path=Path(args.output_path) if args.output_path else rag_store_path
        )
        
        print(f"\n✅ Successfully generated {len(visual_embeddings)} CLIP visual embeddings!")
        print(f"   Final embedding dimension: {visual_embeddings.shape[1]}")
        print(f"   GPU acceleration: {'✅' if torch.cuda.is_available() else '❌'}")
        print(f"   Finnish room types supported: {len(processor.finnish_room_mapping)}")
        print(f"   Used PNG images for optimal performance! 🖼️")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def find_floorplan_directories(dataset_path: Path) -> List[Path]:
    """Find directories containing floor plan files"""
    plan_dirs = []
    
    # Look for numbered directories containing floor plans
    for item in dataset_path.iterdir():
        if item.is_dir() and (item.name.isdigit() or len(item.name) <= 3):
            # Check if directory contains PNG or SVG files
            has_images = any(item.glob("*.png")) or any(item.glob("*.svg"))
            if has_images:
                plan_dirs.append(item)
    
    # Sort by directory name/number
    plan_dirs.sort(key=lambda x: int(x.name) if x.name.isdigit() else x.name)
    return plan_dirs


if __name__ == "__main__":
    main()
