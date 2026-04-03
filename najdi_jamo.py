import sys
import os
import numpy as np
import laspy
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import uniform_filter, generic_filter
from sklearn.cluster import DBSCAN
import shapely.geometry
from shapely.geometry import Point, MultiPoint, box, Polygon
import geopandas as gpd
from collections import defaultdict
import time
from tqdm import tqdm
import configparser
import json
import psutil
import warnings
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
from folium.plugins import LocateControl

# Optional: rasterio for GeoTIFF export
try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Classification constants for better readability
UNCLASSIFIED = 0
UNASSIGNED = 1
GROUND = 2
LOW_VEGETATION = 3
MEDIUM_VEGETATION = 4
HIGH_VEGETATION = 5
BUILDING = 6
NOISE = 7
WATER = 9

@dataclass
class DetectionConfig:
    """Configuration parameters for cave detection"""
    radius: float = 5.0
    depth_threshold: float = 3.0
    min_ground_points: int = 10
    cluster_eps: float = 3.0
    cluster_min_samples: int = 2
    strict_depth_filter: float = 3.0
    include_single_points: bool = False  # NEW OPTION for GUI compatibility
    candidate_classes: List[int] = None
    chunk_size: int = 100000
    max_workers: int = None
    
    def __post_init__(self):
        if self.candidate_classes is None:
            self.candidate_classes = [UNCLASSIFIED, UNASSIGNED, NOISE]
        if self.max_workers is None:
            self.max_workers = min(4, mp.cpu_count())

@dataclass
class ClusterMetrics:
    """Enhanced cluster analysis metrics"""
    cluster_id: int
    point_count: int
    mean_z: float
    min_z: float
    max_z: float
    max_depth: float
    area: float
    perimeter: float
    compactness: float
    aspect_ratio: float
    shape_type: str
    confidence_score: float
    centroid_x: float
    centroid_y: float

class CaveDetector:
    """Enhanced cave entrance detection with GUI compatibility"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.stats = {
            'total_points': 0,
            'ground_points': 0,
            'candidate_points': 0,
            'potential_cave_points': 0,
            'filtered_points': 0,
            'clusters_found': 0,
            'single_points_found': 0,  # For GUI compatibility
            'processing_time': 0,
            'memory_peak': 0
        }
        
    def _monitor_memory(self):
        """Monitor memory usage"""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.stats['memory_peak'] = max(self.stats['memory_peak'], memory_mb)
            return memory_mb
        except Exception:
            return 0
    
    def _load_config_file(self, config_path: str) -> DetectionConfig:
        """Load configuration from file"""
        if not os.path.exists(config_path):
            print(f"[WARNING] Config file not found: {config_path}. Using defaults.")
            return self.config
            
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Update config with file values
            if 'detection' in config:
                section = config['detection']
                self.config.radius = section.getfloat('radius', self.config.radius)
                self.config.depth_threshold = section.getfloat('depth_threshold', self.config.depth_threshold)
                self.config.min_ground_points = section.getint('min_ground_points', self.config.min_ground_points)
                self.config.cluster_eps = section.getfloat('cluster_eps', self.config.cluster_eps)
                self.config.cluster_min_samples = section.getint('cluster_min_samples', self.config.cluster_min_samples)
                self.config.strict_depth_filter = section.getfloat('strict_depth_filter', self.config.strict_depth_filter)
                self.config.include_single_points = section.getboolean('include_single_points', self.config.include_single_points)
                
            print(f"[INFO] Configuration loaded from {config_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}. Using defaults.")
            
        return self.config
    
    def _load_lidar_data(self, laz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and classify LIDAR data with error handling"""
        try:
            print("[INFO] Loading LIDAR file...")
            with tqdm(total=1, desc="[STEP] Load LAS") as pbar:
                las = laspy.read(laz_path)
                pbar.update(1)
            
            points = np.vstack((las.x, las.y, las.z)).T
            classifications = las.classification
            
            self.stats['total_points'] = len(points)
            self._monitor_memory()
            
            print(f"[INFO] Loaded {len(points):,} points")
            return points, classifications, las
            
        except Exception as e:
            print(f"[ERROR] Failed to load LIDAR data: {e}")
            raise
    
    def _classify_points(self, points: np.ndarray, classifications: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify points into ground and candidate categories"""
        print("[INFO] Classifying points...")
        with tqdm(total=1, desc="[STEP] Classify points") as pbar:
            candidate_mask = np.isin(classifications, self.config.candidate_classes)
            ground_mask = classifications == GROUND

            candidate_points = points[candidate_mask]
            ground_points = points[ground_mask]

            self.stats['candidate_points'] = len(candidate_points)
            self.stats['ground_points'] = len(ground_points)

            pbar.update(1)

        print(f"[INFO] Ground points: {len(ground_points):,} | Candidate points: {len(candidate_points):,}")

        if len(ground_points) < 100:
            raise ValueError("Insufficient ground points for analysis")
        if len(candidate_points) < 10:
            raise ValueError("Insufficient candidate points for analysis")

        return candidate_points, ground_points

    def _create_dem(self, ground_points: np.ndarray, resolution: float = 1.0) -> Tuple[np.ndarray, dict]:
        """Create a DEM grid from ground points using binning

        Args:
            ground_points: Array of ground points (N, 3)
            resolution: Grid cell size in meters (default 1.0m)

        Returns:
            dem: 2D array of elevation values (NaN for no data)
            metadata: Dict with bounds and transform info
        """
        print(f"[INFO] Creating DEM at {resolution}m resolution...")

        with tqdm(total=3, desc="[STEP] Build DEM") as pbar:
            # Get bounds
            x_min, x_max = ground_points[:, 0].min(), ground_points[:, 0].max()
            y_min, y_max = ground_points[:, 1].min(), ground_points[:, 1].max()

            # Calculate grid dimensions
            cols = int(np.ceil((x_max - x_min) / resolution)) + 1
            rows = int(np.ceil((y_max - y_min) / resolution)) + 1

            pbar.update(1)

            # Create empty grids for sum and count (for averaging)
            z_sum = np.zeros((rows, cols), dtype=np.float64)
            z_count = np.zeros((rows, cols), dtype=np.int32)

            # Bin points into grid cells
            col_idx = ((ground_points[:, 0] - x_min) / resolution).astype(np.int32)
            row_idx = ((y_max - ground_points[:, 1]) / resolution).astype(np.int32)  # Y flipped for image coords

            # Clip to valid range
            col_idx = np.clip(col_idx, 0, cols - 1)
            row_idx = np.clip(row_idx, 0, rows - 1)

            # Accumulate Z values
            np.add.at(z_sum, (row_idx, col_idx), ground_points[:, 2])
            np.add.at(z_count, (row_idx, col_idx), 1)

            pbar.update(1)

            # Calculate mean elevation (NaN where no points)
            with np.errstate(divide='ignore', invalid='ignore'):
                dem = np.where(z_count > 0, z_sum / z_count, np.nan)

            # Fill small gaps using interpolation (max 3 cell radius)
            dem = self._fill_dem_gaps(dem, max_radius=3)

            pbar.update(1)

        metadata = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'resolution': resolution,
            'rows': rows,
            'cols': cols
        }

        valid_cells = np.sum(~np.isnan(dem))
        print(f"[INFO] DEM created: {cols}x{rows} cells, {valid_cells:,} valid ({100*valid_cells/(rows*cols):.1f}%)")

        return dem, metadata

    def _fill_dem_gaps(self, dem: np.ndarray, max_radius: int = 3) -> np.ndarray:
        """Fill small gaps in DEM using iterative nearest neighbor"""
        filled = dem.copy()

        for r in range(1, max_radius + 1):
            # Find NaN cells
            nan_mask = np.isnan(filled)
            if not np.any(nan_mask):
                break

            # Use uniform filter to get local mean (ignoring NaN properly)
            with np.errstate(invalid='ignore'):
                # Create mask of valid values
                valid_mask = ~np.isnan(filled)
                temp = np.where(valid_mask, filled, 0)

                # Sum and count in neighborhood
                kernel_size = 2 * r + 1
                local_sum = uniform_filter(temp, size=kernel_size, mode='constant', cval=0) * kernel_size**2
                local_count = uniform_filter(valid_mask.astype(float), size=kernel_size, mode='constant', cval=0) * kernel_size**2

                # Fill gaps where we have enough neighbors
                fill_mask = nan_mask & (local_count >= 3)
                filled[fill_mask] = local_sum[fill_mask] / local_count[fill_mask]

        return filled

    def _create_hillshade(self, dem: np.ndarray, metadata: dict,
                          azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
        """Create analytical hillshade from DEM using ESRI/GDAL standard algorithm

        Args:
            dem: 2D elevation array
            metadata: DEM metadata with resolution
            azimuth: Sun azimuth angle in degrees clockwise from North (default 315 = NW)
            altitude: Sun altitude angle in degrees above horizon (default 45)

        Returns:
            hillshade: 2D array with values 0-255
        """
        print(f"[INFO] Creating hillshade (azimuth={azimuth}, altitude={altitude})...")

        resolution = metadata['resolution']

        # Convert angles to radians
        # Azimuth: convert from geographic (clockwise from N) to math (counter-clockwise from E)
        azimuth_rad = np.radians(360.0 - azimuth + 90.0)
        # Zenith angle (90 - altitude)
        zenith_rad = np.radians(90.0 - altitude)

        # Calculate gradients (dz/dx and dz/dy)
        # np.gradient returns (dy, dx) for 2D array - rows change with y, cols with x
        dz_dy, dz_dx = np.gradient(dem, resolution)

        # Calculate slope (steepness) and aspect (direction of steepest descent)
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

        # Aspect: direction of downslope, measured clockwise from North
        # arctan2(dz_dx, -dz_dy) gives aspect where N=0, E=90, S=180, W=270
        aspect_rad = np.arctan2(dz_dx, -dz_dy)

        # Calculate hillshade using standard formula
        # Hillshade = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect)
        hillshade = (np.cos(zenith_rad) * np.cos(slope_rad) +
                    np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad))

        # Normalize to 0-255 range
        hillshade = np.clip(hillshade, 0, 1)
        hillshade = (hillshade * 255).astype(np.uint8)

        # Set NaN areas to middle gray
        hillshade[np.isnan(dem)] = 180

        print(f"[INFO] Hillshade created: {hillshade.shape[1]}x{hillshade.shape[0]} pixels")

        return hillshade

    def _save_dem_outputs(self, dem: np.ndarray, hillshade: np.ndarray,
                          metadata: dict, output_base: str):
        """Save DEM and hillshade as GeoTIFF and PNG"""
        base_name = os.path.splitext(os.path.basename(output_base))[0]
        dem_dir = os.path.join(os.path.dirname(output_base) or ".", "dem")
        os.makedirs(dem_dir, exist_ok=True)

        # Save hillshade as PNG for quick viewing
        hillshade_png = os.path.join(dem_dir, f"{base_name}_hillshade.png")
        try:
            from PIL import Image
            img = Image.fromarray(hillshade)
            img.save(hillshade_png)
            print(f"[INFO] Hillshade PNG saved: {hillshade_png}")
        except ImportError:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            plt.imsave(hillshade_png, hillshade, cmap='gray')
            print(f"[INFO] Hillshade PNG saved: {hillshade_png}")

        # Save as GeoTIFF if rasterio available
        if HAS_RASTERIO:
            # Save DEM GeoTIFF
            dem_tif = os.path.join(dem_dir, f"{base_name}_dem.tif")
            transform = from_bounds(
                metadata['x_min'], metadata['y_min'],
                metadata['x_max'], metadata['y_max'],
                metadata['cols'], metadata['rows']
            )

            with rasterio.open(
                dem_tif, 'w',
                driver='GTiff',
                height=metadata['rows'],
                width=metadata['cols'],
                count=1,
                dtype=dem.dtype,
                crs='EPSG:3794',
                transform=transform,
                nodata=np.nan
            ) as dst:
                dst.write(dem, 1)
            print(f"[INFO] DEM GeoTIFF saved: {dem_tif}")

            # Save hillshade GeoTIFF
            hillshade_tif = os.path.join(dem_dir, f"{base_name}_hillshade.tif")
            with rasterio.open(
                hillshade_tif, 'w',
                driver='GTiff',
                height=metadata['rows'],
                width=metadata['cols'],
                count=1,
                dtype=np.uint8,
                crs='EPSG:3794',
                transform=transform
            ) as dst:
                dst.write(hillshade, 1)
            print(f"[INFO] Hillshade GeoTIFF saved: {hillshade_tif}")

        # Save metadata as JSON
        meta_path = os.path.join(dem_dir, f"{base_name}_dem_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return hillshade_png

    def _analyze_depth_with_dem(self, candidate_points: np.ndarray,
                                 dem: np.ndarray, metadata: dict) -> Tuple[np.ndarray, List[float]]:
        """Analyze depth differences using DEM with neighborhood search

        Uses a circular neighborhood matching the config radius to calculate
        mean ground elevation, similar to the original KDTree approach but faster.

        Args:
            candidate_points: Array of candidate points (N, 3)
            dem: 2D elevation array
            metadata: DEM metadata

        Returns:
            potential_cave_points: Array of points below threshold
            depth_differences: List of depth values
        """
        print("[INFO] Analyzing depth against DEM...")

        resolution = metadata['resolution']
        x_min = metadata['x_min']
        y_max = metadata['y_max']
        rows, cols = metadata['rows'], metadata['cols']

        # Calculate neighborhood size based on radius
        radius_cells = int(np.ceil(self.config.radius / resolution))

        # Pre-compute circular kernel mask
        y_kernel, x_kernel = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
        kernel_mask = x_kernel**2 + y_kernel**2 <= radius_cells**2

        potential_cave_points = []
        depth_differences = []

        # Convert point coordinates to grid indices
        col_idx = ((candidate_points[:, 0] - x_min) / resolution).astype(np.int32)
        row_idx = ((y_max - candidate_points[:, 1]) / resolution).astype(np.int32)

        with tqdm(total=len(candidate_points), desc="[PROCESS] DEM depth check") as pbar:
            # Process in batches for progress bar
            batch_size = 50000
            for i in range(0, len(candidate_points), batch_size):
                batch_end = min(i + batch_size, len(candidate_points))
                batch_pts = candidate_points[i:batch_end]
                batch_cols = col_idx[i:batch_end]
                batch_rows = row_idx[i:batch_end]

                for pt, c, r in zip(batch_pts, batch_cols, batch_rows):
                    # Check if center is within bounds
                    if not (radius_cells <= c < cols - radius_cells and
                            radius_cells <= r < rows - radius_cells):
                        # Edge case: use simple lookup
                        if 0 <= c < cols and 0 <= r < rows:
                            ground_z = dem[r, c]
                            if not np.isnan(ground_z):
                                depth_diff = ground_z - pt[2]
                                if depth_diff >= self.config.depth_threshold:
                                    potential_cave_points.append(pt)
                                    depth_differences.append(depth_diff)
                        continue

                    # Extract neighborhood
                    neighborhood = dem[r-radius_cells:r+radius_cells+1,
                                      c-radius_cells:c+radius_cells+1]

                    # Apply circular mask and get valid values
                    valid_values = neighborhood[kernel_mask]
                    valid_values = valid_values[~np.isnan(valid_values)]

                    # Check minimum ground points requirement
                    if len(valid_values) < self.config.min_ground_points:
                        continue

                    # Calculate mean ground elevation in neighborhood
                    ground_z = np.mean(valid_values)
                    depth_diff = ground_z - pt[2]

                    if depth_diff >= self.config.depth_threshold:
                        potential_cave_points.append(pt)
                        depth_differences.append(depth_diff)

                pbar.update(batch_end - i)

        if potential_cave_points:
            potential_cave_points = np.array(potential_cave_points)
            self.stats['potential_cave_points'] = len(potential_cave_points)
            print(f"[INFO] Found {len(potential_cave_points):,} potential cave points")
        else:
            potential_cave_points = np.array([]).reshape(0, 3)

        return potential_cave_points, depth_differences
    
    def _process_chunk(self, chunk_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        """Process a chunk of candidate points"""
        candidate_chunk, ground_points = chunk_data
        chunk_cave_points = []
        chunk_depths = []
        
        # Build KDTree for this chunk processing
        ground_tree = cKDTree(ground_points[:, :2])
        
        for pt in candidate_chunk:
            try:
                idx = ground_tree.query_ball_point(pt[:2], r=self.config.radius)
                if len(idx) >= self.config.min_ground_points:
                    # Get ground Z values using the indices
                    local_ground_z = ground_points[idx, 2]
                    ground_mean_z = np.mean(local_ground_z)
                    depth_diff = ground_mean_z - pt[2]
                    
                    if depth_diff >= self.config.depth_threshold:
                        chunk_cave_points.append(pt)
                        chunk_depths.append(depth_diff)
            except Exception as e:
                continue  # Skip problematic points
                
        return chunk_cave_points, chunk_depths
    
    def _analyze_depth_parallel(self, candidate_points: np.ndarray, ground_points: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Analyze depth differences using parallel processing"""
        print("[INFO] Analyzing candidate points...")
        
        # For smaller datasets, use original sequential approach for better performance
        if len(candidate_points) < 50000:
            return self._analyze_depth_sequential(candidate_points, ground_points)
        
        # Split candidate points into chunks
        chunks = []
        for i in range(0, len(candidate_points), self.config.chunk_size):
            chunk = candidate_points[i:i + self.config.chunk_size]
            chunks.append((chunk, ground_points))
        
        potential_cave_points = []
        depth_differences = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            with tqdm(total=len(chunks), desc="[PROCESS] Checking depth") as pbar:
                futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]
                
                for future in futures:
                    try:
                        chunk_points, chunk_depths = future.result()
                        potential_cave_points.extend(chunk_points)
                        depth_differences.extend(chunk_depths)
                    except Exception as e:
                        print(f"[WARNING] Chunk processing failed: {e}")
                    finally:
                        pbar.update(1)
        
        if potential_cave_points:
            potential_cave_points = np.array(potential_cave_points)
            self.stats['potential_cave_points'] = len(potential_cave_points)
            print(f"[INFO] Found {len(potential_cave_points):,} potential cave points")
        else:
            potential_cave_points = np.array([]).reshape(0, 3)
            
        return potential_cave_points, depth_differences
    
    def _analyze_depth_sequential(self, candidate_points: np.ndarray, ground_points: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Original sequential depth analysis for better performance on smaller datasets"""
        print("[INFO] Using sequential processing for better performance...")
        
        # Build KDTree for ground points (original approach)
        ground_tree = cKDTree(ground_points[:, :2])
        
        potential_cave_points = []
        depth_differences = []
        
        with tqdm(total=len(candidate_points), desc="[PROCESS] Checking depth") as pbar:
            for pt in candidate_points:
                pbar.update(1)
                try:
                    idx = ground_tree.query_ball_point(pt[:2], r=self.config.radius)
                    if len(idx) >= self.config.min_ground_points:
                        local_ground_z = ground_points[idx, 2]
                        ground_mean_z = np.mean(local_ground_z)
                        depth_diff = ground_mean_z - pt[2]
                        if depth_diff >= self.config.depth_threshold:
                            potential_cave_points.append(pt)
                            depth_differences.append(depth_diff)
                except Exception as e:
                    continue  # Skip problematic points
        
        if potential_cave_points:
            potential_cave_points = np.array(potential_cave_points)
            self.stats['potential_cave_points'] = len(potential_cave_points)
            print(f"[INFO] Found {len(potential_cave_points):,} potential cave points")
        else:
            potential_cave_points = np.array([]).reshape(0, 3)
            
        return potential_cave_points, depth_differences
    
    def _calculate_cluster_metrics(self, points: List[np.ndarray], depths: List[float], cluster_id: int) -> ClusterMetrics:
        """Calculate comprehensive cluster metrics including single points"""
        coords_2d = np.array([[pt[0], pt[1]] for pt in points])
        coords_3d = np.array(points)
        
        # Basic statistics
        mean_z = np.mean(coords_3d[:, 2])
        min_z = np.min(coords_3d[:, 2])
        max_z = np.max(coords_3d[:, 2])
        max_depth = np.max(depths)
        
        # Handle single points differently
        if len(points) == 1:
            area = 0
            perimeter = 0
            compactness = 0
            aspect_ratio = 1.0
            shape_type = "posamezna_točka"
            confidence_score = min(max_depth / 10.0, 0.7)  # Cap single points at 0.7
            
        elif len(points) == 2:
            area = 0
            perimeter = np.linalg.norm(coords_2d[1] - coords_2d[0])
            compactness = 0
            aspect_ratio = 1.0
            shape_type = "par_točk"
            depth_score = min(max_depth / 10.0, 1.0)
            size_score = 0.4  # Lower for pairs
            confidence_score = (depth_score + size_score) / 2.0
            
        else:
            # Existing logic for larger clusters
            try:
                if len(coords_2d) >= 3:
                    hull = ConvexHull(coords_2d)
                    area = hull.volume  # In 2D, volume is area
                    perimeter = 0
                    for simplex in hull.simplices:
                        p1, p2 = coords_2d[simplex]
                        perimeter += np.linalg.norm(p2 - p1)
                else:
                    area = 0
                    perimeter = 0
            except:
                area = 0
                perimeter = 0
            
            # Compactness (isoperimetric quotient)
            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Aspect ratio
            if len(coords_2d) > 1:
                x_range = np.ptp(coords_2d[:, 0])
                y_range = np.ptp(coords_2d[:, 1])
                aspect_ratio = max(x_range, y_range) / max(min(x_range, y_range), 0.0001)
            else:
                aspect_ratio = 1.0
            
            # Shape classification
            if aspect_ratio < 1.5:
                shape_type = "okrogla"
            elif aspect_ratio < 3.0:
                shape_type = "ovalna"
            else:
                shape_type = "podolgovata"
            
            # Full confidence calculation for clusters 3+
            depth_score = min(max_depth / 10.0, 1.0)
            size_score = min(len(points) / 20.0, 1.0)
            shape_score = max(0, 1.0 - (aspect_ratio - 1.0) / 4.0)
            confidence_score = (depth_score + size_score + shape_score) / 3.0
        
        # Centroid
        centroid = np.mean(coords_2d, axis=0)
        
        return ClusterMetrics(
            cluster_id=cluster_id,
            point_count=len(points),
            mean_z=mean_z,
            min_z=min_z,
            max_z=max_z,
            max_depth=max_depth,
            area=area,
            perimeter=perimeter,
            compactness=compactness,
            aspect_ratio=aspect_ratio,
            shape_type=shape_type,
            confidence_score=confidence_score,
            centroid_x=centroid[0],
            centroid_y=centroid[1]
        )
    
    def _cluster_and_filter(self, potential_cave_points: np.ndarray, depth_differences: List[float]) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Cluster points and apply filtering with optional single point inclusion"""
        if len(potential_cave_points) < 1:
            print("[INFO] No points to cluster.")
            return None, None
        
        print("[INFO] Clustering potential points...")
        with tqdm(total=1, desc="[STEP] Clustering") as pbar:
            clustering = DBSCAN(
                eps=self.config.cluster_eps, 
                min_samples=self.config.cluster_min_samples
            ).fit(potential_cave_points[:, :2])
            labels = clustering.labels_
            pbar.update(1)
        
        # Filter by depth and handle single points based on config
        filtered_points = []
        filtered_labels = []
        filtered_depths = []
        
        noise_counter = 10000  # Start from high number for single points
        
        for pt, lbl, dz in zip(potential_cave_points, labels, depth_differences):
            if dz >= self.config.strict_depth_filter:
                if lbl == -1:  # Noise/single point
                    if self.config.include_single_points:
                        filtered_points.append(pt)
                        filtered_labels.append(noise_counter)
                        filtered_depths.append(dz)
                        noise_counter += 1
                    # Else: skip single points
                else:  # Clustered point
                    filtered_points.append(pt)
                    filtered_labels.append(lbl)
                    filtered_depths.append(dz)
        
        self.stats['filtered_points'] = len(filtered_points)
        
        # Report statistics
        if self.config.include_single_points:
            single_points = sum(1 for lbl in filtered_labels if lbl >= 10000)
            cluster_points = len(filtered_labels) - single_points
            real_clusters = len(set(lbl for lbl in filtered_labels if lbl < 10000))
            self.stats['single_points_found'] = single_points
            print(f"[INFO] Results: {real_clusters} multi-point clusters, {single_points} single points")
        else:
            real_clusters = len(set(filtered_labels)) if filtered_labels else 0
            self.stats['single_points_found'] = 0
            print(f"[INFO] Results: {real_clusters} multi-point clusters (single points excluded)")
        
        if not filtered_points:
            print("[INFO] No points passed filtering.")
            return None, None
        
        # Create point GeoDataFrame
        gdf_points = gpd.GeoDataFrame({
            "cluster": filtered_labels,
            "X": [pt[0] for pt in filtered_points],
            "Y": [pt[1] for pt in filtered_points],
            "Z": [pt[2] for pt in filtered_points],
            "deltaZ": filtered_depths
        }, geometry=[Point(pt[0], pt[1]) for pt in filtered_points], crs="EPSG:3794")
        
        # Generate enhanced cluster summaries
        print("[INFO] Generating enhanced cluster metrics...")
        cluster_data = defaultdict(list)
        for pt, lbl, dz in zip(filtered_points, filtered_labels, filtered_depths):
            cluster_data[lbl].append((pt, dz))
        
        cluster_summaries = []
        for lbl, pts in cluster_data.items():
            coords = [pt for pt, _ in pts]
            depths = [dz for _, dz in pts]
            
            metrics = self._calculate_cluster_metrics(coords, depths, lbl)
            cluster_summaries.append(asdict(metrics))
        
        # Create cluster GeoDataFrame
        gdf_clusters = gpd.GeoDataFrame([
            {
                **summary,
                "geometry": Point(summary["centroid_x"], summary["centroid_y"])
            }
            for summary in cluster_summaries
        ], crs="EPSG:3794")
        
        self.stats['clusters_found'] = len(gdf_clusters)
        print(f"[INFO] Generated {len(gdf_clusters)} cluster summaries")
        
        return gdf_points, gdf_clusters
    
    def _create_enhanced_map(self, gdf_points: gpd.GeoDataFrame, gdf_clusters: gpd.GeoDataFrame, output_path: str):
        """Create enhanced interactive map with better visualization"""
        print("[INFO] Creating enhanced HTML map...")
        
        # Convert to WGS84
        gdf_points_wgs = gdf_points.to_crs(epsg=4326)
        gdf_clusters_wgs = gdf_clusters.to_crs(epsg=4326)
        
        # Create map
        center_lat = gdf_points_wgs.geometry.y.mean()
        center_lon = gdf_points_wgs.geometry.x.mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
        
        # Add base layers
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        folium.TileLayer(
            tiles='https://gis.geo-zs.si/image/rest/services/TOPO/Hillshade/ImageServer/tile/{z}/{y}/{x}',
            attr='GeoZS Hillshade',
            name='Hillshade GeoZS',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Color mapping for depth
        cmap = cm.get_cmap('plasma', lut=None)
        norm = colors.Normalize(vmin=gdf_points['deltaZ'].min(), vmax=gdf_points['deltaZ'].max())
        
        # Add point markers with enhanced popups
        for idx, row in gdf_points_wgs.iterrows():
            popup_html = f"""
            <b>Cluster:</b> {row['cluster']}<br>
            <b>Depth:</b> {row['deltaZ']:.2f} m<br>
            <b>Coordinates:</b> {row['X']:.2f}, {row['Y']:.2f}<br>
            <b>Elevation:</b> {row['Z']:.2f} m
            """
            color = colors.to_hex(cmap(norm(row['deltaZ'])))
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
        
        # Add cluster markers with comprehensive information
        for idx, row in gdf_clusters_wgs.iterrows():
            popup_html = f"""
            <b>Cluster ID:</b> {row['cluster_id']}<br>
            <b>Points:</b> {row['point_count']}<br>
            <b>Type:</b> {row['shape_type']}<br>
            <b>Confidence:</b> {row['confidence_score']:.2f}<br>
            <b>Max Depth:</b> {row['max_depth']:.2f} m<br>
            <b>Area:</b> {row['area']:.2f} m²<br>
            <b>Aspect Ratio:</b> {row['aspect_ratio']:.2f}<br>
            <b>Compactness:</b> {row['compactness']:.3f}
            """
            
            # Color code by confidence and type
            if row['shape_type'] == 'posamezna_točka':
                icon_color = 'purple'
                icon_symbol = 'record'
            elif row['shape_type'] == 'par_točk':
                icon_color = 'blue'
                icon_symbol = 'pause'
            elif row['confidence_score'] > 0.7:
                icon_color = 'green'
                icon_symbol = 'info-sign'
            elif row['confidence_score'] > 0.5:
                icon_color = 'orange'
                icon_symbol = 'info-sign'
            else:
                icon_color = 'red'
                icon_symbol = 'info-sign'
                
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                icon=folium.Icon(color=icon_color, icon=icon_symbol),
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
        
        # Add known caves if available
        self._add_known_caves(m, gdf_clusters_wgs)
        
        # Enhanced legend with single point info
        single_count = len(gdf_clusters[gdf_clusters['shape_type'] == 'posamezna_točka'])
        pair_count = len(gdf_clusters[gdf_clusters['shape_type'] == 'par_točk'])
        cluster_count = len(gdf_clusters) - single_count - pair_count
        
        legend_html = f'''
        <div style="position: fixed; 
                   bottom: 50px; left: 50px; width: 220px; height: 200px; 
                   background-color: white; z-index:9999; font-size:12px; 
                   border:2px solid grey; padding: 10px; border-radius: 5px;">
        <b>Cave Detection Results</b><br>
        <b>Depth (deltaZ):</b><br>
        <i>min:</i> {gdf_points['deltaZ'].min():.2f} m<br>
        <i>max:</i> {gdf_points['deltaZ'].max():.2f} m<br>
        <div style="background: linear-gradient(to right, #440154, #fde725); height: 10px; margin: 5px 0;"></div>
        <b>Result Types:</b><br>
        🟢 Clusters 3+ (High conf.) 🟠 Medium<br>
        🔴 Low conf. 🟣 Single points<br>
        🔵 Point pairs<br>
        <b>Counts:</b><br>
        Clusters: {cluster_count} | Singles: {single_count} | Pairs: {pair_count}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add controls
        LocateControl(auto_start=True).add_to(m)
        folium.LayerControl().add_to(m)
        
        # Save map
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        m.save(output_path)
        print(f"[INFO] Enhanced map saved to: {output_path}")
    
    def _add_known_caves(self, map_obj: folium.Map, gdf_clusters_wgs: gpd.GeoDataFrame):
        """Add known caves to the map if available"""
        _base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        known_caves_path = os.path.join(_base, "jame", "SLO_JAME_2024.geojson")
        if not os.path.exists(known_caves_path):
            known_caves_path = os.path.join("jame", "SLO_JAME_2024.geojson")
        if not os.path.exists(known_caves_path):
            print("[WARNING] No known caves geojson found in /jame folder.")
            return
        
        try:
            known_caves = gpd.read_file(known_caves_path)
            known_caves = known_caves[known_caves.geometry.notnull()]
            known_caves = known_caves.to_crs(epsg=4326)
            
            # Create bounding box around detections
            bounds = gdf_clusters_wgs.total_bounds
            expanded_bounds = [bounds[0] - 0.003, bounds[1] - 0.003, 
                             bounds[2] + 0.003, bounds[3] + 0.003]
            bbox = box(*expanded_bounds)
            caves_nearby = known_caves[known_caves.intersects(bbox)]
            
            for idx, row in caves_nearby.iterrows():
                if isinstance(row.geometry, Point):
                    popup_html = f"""
                    <b>Known Cave</b><br>
                    <b>Cat. No.:</b> {row.get('kat_št_', 'n/a')}<br>
                    <b>Name:</b> {row.get('Ime', 'n/a')}<br>
                    <b>Depth:</b> {row.get('Globina', 'n/a')}<br>
                    <b>Length:</b> {row.get('Dolžina', 'n/a')}
                    """
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        icon=folium.Icon(color='red', icon='flag'),
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(map_obj)
            
            print(f"[INFO] Added {len(caves_nearby)} known caves to the map.")
            
        except Exception as e:
            print(f"[WARNING] Failed to load known caves: {e}")
    
    def _save_outputs(self, gdf_points: gpd.GeoDataFrame, gdf_clusters: gpd.GeoDataFrame, output_base: str):
        """Save all outputs with enhanced metadata"""
        base_name = os.path.splitext(os.path.basename(output_base))[0]
        
        # Save point shapefile
        points_path = os.path.join("shp", f"{base_name}_cave_points.shp")
        os.makedirs(os.path.dirname(points_path), exist_ok=True)
        gdf_points.to_file(points_path)
        print(f"[INFO] Point shapefile saved: {points_path}")
        
        # Save cluster shapefile
        clusters_path = os.path.join("cluster", f"{base_name}_clusters.shp")
        os.makedirs(os.path.dirname(clusters_path), exist_ok=True)
        gdf_clusters.to_file(clusters_path)
        print(f"[INFO] Cluster shapefile saved: {clusters_path}")
        
        # Save enhanced map
        map_path = os.path.join("map", f"{base_name}_map.html")
        self._create_enhanced_map(gdf_points, gdf_clusters, map_path)
        
        # Save processing statistics
        stats_path = os.path.join("stats", f"{base_name}_stats.json")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        
        enhanced_stats = {
            **self.stats,
            'config': asdict(self.config),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'high_confidence_clusters': len(gdf_clusters[gdf_clusters['confidence_score'] > 0.7]),
            'medium_confidence_clusters': len(gdf_clusters[gdf_clusters['confidence_score'] > 0.5]),
            'avg_cluster_confidence': float(gdf_clusters['confidence_score'].mean()) if len(gdf_clusters) > 0 else 0.0,
            'max_depth_found': float(gdf_points['deltaZ'].max()) if len(gdf_points) > 0 else 0.0,
            'avg_depth': float(gdf_points['deltaZ'].mean()) if len(gdf_points) > 0 else 0.0,
            'single_points': int(self.stats['single_points_found']),
            'pair_points': len(gdf_clusters[gdf_clusters['shape_type'] == 'par_točk']) if len(gdf_clusters) > 0 else 0,
            'cluster_3plus': len(gdf_clusters[~gdf_clusters['shape_type'].isin(['posamezna_točka', 'par_točk'])]) if len(gdf_clusters) > 0 else 0
        }
        
        with open(stats_path, 'w') as f:
            json.dump(enhanced_stats, f, indent=2)
        print(f"[INFO] Processing statistics saved: {stats_path}")
    
    def detect_cave_entrances(self, laz_path: str, output_base: str, config_path: str = None, return_raw_data: bool = False):
        """Main detection pipeline using DEM-based depth analysis

        Args:
            laz_path: Path to input LAZ/LAS file
            output_base: Base path for output files
            config_path: Optional path to configuration file
            return_raw_data: If True, return dict with raw data for GUI visualization

        Returns:
            If return_raw_data is False: bool indicating success
            If return_raw_data is True: tuple (success: bool, data: dict) where data contains:
                - ground_points: np.ndarray of ground points (N, 3)
                - gdf_points: GeoDataFrame of detected cave points
                - gdf_clusters: GeoDataFrame of cluster summaries
                - potential_cave_points: np.ndarray of unfiltered cave candidates
                - depth_differences: list of depth values for each candidate
                - dem: 2D elevation array
                - dem_metadata: DEM bounds and resolution info
                - hillshade_path: Path to saved hillshade PNG
        """
        start_time = time.time()

        try:
            # Load configuration if provided
            if config_path and os.path.exists(config_path):
                self._load_config_file(config_path)

            print(f"[INFO] Starting cave detection with config: {asdict(self.config)}")

            # Load and process data
            points, classifications, las_data = self._load_lidar_data(laz_path)
            candidate_points, ground_points = self._classify_points(points, classifications)

            # Create DEM from ground points (0.5m resolution for better hillshade quality)
            dem, dem_metadata = self._create_dem(ground_points, resolution=0.5)

            # Create hillshade for visualization
            hillshade = self._create_hillshade(dem, dem_metadata)

            # Save DEM outputs
            hillshade_path = self._save_dem_outputs(dem, hillshade, dem_metadata, output_base)

            # Analyze depth using DEM (much faster than KDTree)
            potential_cave_points, depth_differences = self._analyze_depth_with_dem(
                candidate_points, dem, dem_metadata
            )

            if len(potential_cave_points) == 0:
                print("[INFO] No potential cave points detected.")
                if return_raw_data:
                    return False, {
                        'ground_points': ground_points,
                        'gdf_points': None,
                        'gdf_clusters': None,
                        'potential_cave_points': potential_cave_points,
                        'depth_differences': depth_differences,
                        'dem': dem,
                        'dem_metadata': dem_metadata,
                        'hillshade_path': hillshade_path
                    }
                return False

            # Cluster and filter
            gdf_points, gdf_clusters = self._cluster_and_filter(
                potential_cave_points, depth_differences
            )

            if gdf_points is None or gdf_clusters is None:
                print("[INFO] No valid clusters found after filtering.")
                if return_raw_data:
                    return False, {
                        'ground_points': ground_points,
                        'gdf_points': None,
                        'gdf_clusters': None,
                        'potential_cave_points': potential_cave_points,
                        'depth_differences': depth_differences,
                        'dem': dem,
                        'dem_metadata': dem_metadata,
                        'hillshade_path': hillshade_path
                    }
                return False

            # Save all outputs
            self._save_outputs(gdf_points, gdf_clusters, output_base)

            # Final statistics
            self.stats['processing_time'] = time.time() - start_time
            self._monitor_memory()

            print(f"\n[DONE] Cave detection completed successfully!")
            print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
            print(f"Memory peak: {self.stats['memory_peak']:.1f} MB")
            print(f"Total results: {self.stats['clusters_found']} (including {self.stats['single_points_found']} singles)")
            high_conf_count = len(gdf_clusters[gdf_clusters['confidence_score'] > 0.7]) if len(gdf_clusters) > 0 else 0
            print(f"High confidence clusters: {high_conf_count}")

            if return_raw_data:
                return True, {
                    'ground_points': ground_points,
                    'gdf_points': gdf_points,
                    'gdf_clusters': gdf_clusters,
                    'potential_cave_points': potential_cave_points,
                    'depth_differences': depth_differences,
                    'dem': dem,
                    'dem_metadata': dem_metadata,
                    'hillshade_path': hillshade_path
                }
            return True

        except Exception as e:
            print(f"[ERROR] Cave detection failed: {e}")
            import traceback
            traceback.print_exc()
            if return_raw_data:
                return False, None
            return False

def create_default_config(config_path: str = "cave_config.ini"):
    """Create a default configuration file"""
    config = configparser.ConfigParser()
    config['detection'] = {
        'radius': '5.0',
        'depth_threshold': '2.0', 
        'min_ground_points': '10',
        'cluster_eps': '3.0',
        'cluster_min_samples': '2',
        'strict_depth_filter': '3.0',
        'include_single_points': 'False'
    }
    
    with open(config_path, 'w') as f:
        config.write(f)
    print(f"[INFO] Default configuration created: {config_path}")

def main():
    """Enhanced main function with better argument handling"""
    if len(sys.argv) < 2:
        print("Usage: python najdi_jamo.py <input_file.laz> [config_file.ini]")
        print("       python najdi_jamo.py --create-config [config_file.ini]")
        sys.exit(1)
    
    # Handle config creation
    if sys.argv[1] == "--create-config":
        config_path = sys.argv[2] if len(sys.argv) > 2 else "cave_config.ini"
        create_default_config(config_path)
        return
    
    # Normal processing
    input_filename = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    input_path = os.path.join("laz", input_filename)
    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)
    
    # Create detector and run
    detector = CaveDetector()
    
    output_base = os.path.splitext(input_filename)[0]
    success = detector.detect_cave_entrances(input_path, output_base, config_path)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 