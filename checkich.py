# Modern Cave Detection GUI using PySide6/Qt6
# Enhanced with embedded map, tabbed layout, and improved 3D rendering

import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFrame, QLabel, QPushButton, QLineEdit, QComboBox,
    QCheckBox, QSlider, QProgressBar, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QStatusBar,
    QTabWidget, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QObject, QThread, QUrl, Slot, QTimer
from PySide6.QtGui import QFont, QColor, QAction, QPalette
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel

# Matplotlib for 3D rendering
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Folium for map generation
import folium
from folium.plugins import LocateControl

# Import detection module
try:
    from najdi_jamo import CaveDetector, DetectionConfig
except ImportError:
    print("Error: najdi_jamo.py not found")
    sys.exit(1)

# Optional imports
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

SETTINGS_FILE = "gui_settings.json"

# Soft blue/grey theme - easier on the eyes
SOFT_STYLE = """
QMainWindow, QWidget {
    background-color: #2d3748;
    color: #e2e8f0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #4a5568;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px;
    background-color: #3d4a5c;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #90cdf4;
    font-weight: bold;
}
QPushButton {
    background-color: #4a6fa5;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    color: #e2e8f0;
    font-weight: bold;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #5a7fb5;
}
QPushButton:pressed {
    background-color: #3a5f95;
}
QPushButton:disabled {
    background-color: #4a5568;
    color: #718096;
}
QPushButton#startButton {
    background-color: #48bb78;
    color: #1a202c;
    min-height: 40px;
    font-size: 11pt;
}
QPushButton#startButton:hover {
    background-color: #68d391;
}
QPushButton#startButton:disabled {
    background-color: #4a5568;
    color: #718096;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #4a5568;
    border: 1px solid #5a6578;
    border-radius: 4px;
    padding: 6px;
    color: #e2e8f0;
    min-height: 20px;
    selection-background-color: #4a6fa5;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #90cdf4;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #4a5568;
    border: 1px solid #5a6578;
    selection-background-color: #4a6fa5;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #4a5568;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #90cdf4;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #4a6fa5;
    border-radius: 2px;
}
QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #4a5568;
    height: 6px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #4a6fa5;
    border-radius: 4px;
}
QTableWidget {
    background-color: #3d4a5c;
    alternate-background-color: #455264;
    border: 1px solid #4a5568;
    border-radius: 6px;
    gridline-color: #4a5568;
    font-size: 9pt;
}
QTableWidget::item {
    padding: 4px;
    border: none;
}
QTableWidget::item:selected {
    background-color: #4a6fa5;
}
QHeaderView::section {
    background-color: #4a5568;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #5a6578;
    font-weight: bold;
    font-size: 9pt;
    color: #e2e8f0;
}
QTextEdit {
    background-color: #2d3748;
    border: 1px solid #4a5568;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    color: #a0aec0;
}
QTabWidget::pane {
    border: 1px solid #4a5568;
    border-radius: 6px;
    background-color: #3d4a5c;
    top: -1px;
}
QTabBar::tab {
    background-color: #4a5568;
    color: #a0aec0;
    padding: 10px 24px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background-color: #3d4a5c;
    color: #90cdf4;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background-color: #5a6578;
}
QScrollBar:vertical {
    background-color: #3d4a5c;
    width: 10px;
    border-radius: 5px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #5a6578;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #718096;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background-color: #3d4a5c;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #5a6578;
    border-radius: 5px;
    min-width: 30px;
}
QSplitter::handle {
    background-color: #4a5568;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}
QStatusBar {
    background-color: #3d4a5c;
    color: #90cdf4;
    font-size: 9pt;
    border-top: 1px solid #4a5568;
}
QLabel#titleLabel {
    font-size: 18pt;
    font-weight: bold;
    color: #90cdf4;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #5a6578;
    background-color: #4a5568;
}
QCheckBox::indicator:checked {
    background-color: #4a6fa5;
    border-color: #4a6fa5;
}
QCheckBox::indicator:hover {
    border-color: #90cdf4;
}
QToolBar {
    background-color: #3d4a5c;
    border: none;
    spacing: 4px;
    padding: 4px;
}
QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 4px;
}
QToolButton:hover {
    background-color: #4a5568;
}
QToolButton:pressed {
    background-color: #5a6578;
}
"""


@dataclass
class ProcessingResults:
    """Store detection results for visualization"""
    gdf_points: Optional[object] = None
    gdf_clusters: Optional[object] = None
    ground_points: Optional[np.ndarray] = None
    potential_cave_points: Optional[np.ndarray] = None
    depth_differences: Optional[list] = None
    dem: Optional[np.ndarray] = None
    dem_metadata: Optional[dict] = None
    hillshade_path: Optional[str] = None


class WorkerSignals(QObject):
    """Signals for worker thread communication"""
    started = Signal()
    progress = Signal(int, str)
    finished = Signal(bool, object)
    error = Signal(str)
    log = Signal(str)


class DetectionWorker(QThread):
    """Worker thread for cave detection processing"""

    def __init__(self, input_file, output_base, config):
        super().__init__()
        self.input_file = input_file
        self.output_base = output_base
        self.config = config
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.started.emit()
            self.signals.log.emit(f"Processing: {os.path.basename(self.input_file)}")

            detector = CaveDetector(self.config)
            result = detector.detect_cave_entrances(
                self.input_file,
                self.output_base,
                return_raw_data=True
            )

            if isinstance(result, tuple):
                success, data = result
            else:
                success = result
                data = None

            self.signals.finished.emit(success, data)

        except Exception as e:
            self.signals.error.emit(str(e))


class MapBridge(QObject):
    """Bridge for JavaScript to Python communication"""
    show_3d = Signal(int)

    @Slot(int)
    def requestShow3D(self, cluster_id):
        self.show_3d.emit(cluster_id)


class InteractiveMapWidget(QWidget):
    """Embedded interactive map using QWebEngineView"""

    cluster_selected = Signal(int)
    show_3d_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cluster_data = {}
        self.gdf_clusters = None
        self.known_caves = None
        self.data_extent = None  # (min_x, min_y, max_x, max_y) in EPSG:3794
        self.current_html = None
        self.hillshade_path = None
        self.dem_metadata = None
        self.setup_ui()
        self.load_known_caves()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Web view for map with WebChannel for JS-Python communication
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(300)

        # Setup WebChannel for communication
        self.bridge = MapBridge()
        self.bridge.show_3d.connect(self.on_show_3d_clicked)
        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        layout.addWidget(self.web_view, stretch=1)

        # Control bar
        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("Podlaga:"))
        self.tile_combo = QComboBox()
        self.tile_combo.addItems(["OpenStreetMap", "Satelit", "Topografska", "Hillshade (lokalno)"])
        self.tile_combo.setMaximumWidth(140)
        self.tile_combo.currentIndexChanged.connect(self.regenerate_map)
        controls.addWidget(self.tile_combo)

        self.show_known_cb = QCheckBox("Znane jame")
        self.show_known_cb.setChecked(True)
        self.show_known_cb.setToolTip("Prikaži jame iz katastra")
        self.show_known_cb.stateChanged.connect(self.regenerate_map)
        controls.addWidget(self.show_known_cb)

        controls.addStretch()

        self.refresh_btn = QPushButton("Osveži")
        self.refresh_btn.setMaximumWidth(80)
        self.refresh_btn.clicked.connect(self.regenerate_map)
        controls.addWidget(self.refresh_btn)

        layout.addLayout(controls)

        # Show placeholder
        self.show_placeholder()

    def load_known_caves(self):
        """Load known caves from geojson file"""
        geojson_path = os.path.join(os.path.dirname(__file__), "jame", "SLO_JAME_2024.geojson")
        if os.path.exists(geojson_path) and HAS_GEOPANDAS:
            try:
                self.known_caves = gpd.read_file(geojson_path)
                print(f"Loaded {len(self.known_caves)} known caves")
            except Exception as e:
                print(f"Failed to load known caves: {e}")
                self.known_caves = None

    def on_show_3d_clicked(self, cluster_id):
        """Handle Show 3D button click from map popup"""
        self.show_3d_requested.emit(cluster_id)

    def set_data_extent(self, min_x, min_y, max_x, max_y):
        """Set the extent of processed data for filtering known caves"""
        self.data_extent = (min_x, min_y, max_x, max_y)

    def set_hillshade(self, hillshade_path: str, dem_metadata: dict):
        """Set the local hillshade for use as basemap"""
        self.hillshade_path = hillshade_path
        self.dem_metadata = dem_metadata

    def show_placeholder(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    background: #3d4a5c;
                    color: #a0aec0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    font-family: 'Segoe UI', Arial, sans-serif;
                }
                .msg { text-align: center; }
                h2 { color: #90cdf4; margin-bottom: 10px; }
                p { color: #a0aec0; }
            </style>
        </head>
        <body>
            <div class="msg">
                <h2>Interaktivna karta</h2>
                <p>Poženi detekcijo, da se prikaže karta</p>
            </div>
        </body>
        </html>
        """
        self.web_view.setHtml(html)

    def update_map(self, gdf_clusters, gdf_points=None):
        """Store cluster data and generate map"""
        self.gdf_clusters = gdf_clusters
        self.regenerate_map()

    def regenerate_map(self):
        """Generate and display interactive map with clusters"""
        import sys
        tile_idx = self.tile_combo.currentIndex()
        sys.stderr.write(f"[MAP] regenerate_map called, tile_index={tile_idx}\n")
        sys.stderr.flush()

        if self.gdf_clusters is None or len(self.gdf_clusters) == 0:
            self.show_placeholder()
            return

        self.cluster_data = {}

        # Transform to WGS84
        if not HAS_GEOPANDAS:
            self.show_placeholder()
            return

        try:
            gdf_wgs = self.gdf_clusters.to_crs(epsg=4326)
        except Exception as e:
            print(f"CRS transform error: {e}")
            self.show_placeholder()
            return

        # Get selected tile layer
        tile_index = self.tile_combo.currentIndex()

        # Create map centered on clusters
        center_lat = gdf_wgs.geometry.y.mean()
        center_lon = gdf_wgs.geometry.x.mean()

        # Create base map based on selection
        if tile_index == 0:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='OpenStreetMap')
        elif tile_index == 1:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google Satellite',
                name='Satellite'
            ).add_to(m)
        elif tile_index == 2:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)
            folium.TileLayer(
                tiles='https://tile.opentopomap.org/{z}/{x}/{y}.png',
                attr='OpenTopoMap',
                name='Topographic',
                max_zoom=17
            ).add_to(m)
        elif tile_index == 3:
            # Local hillshade - use simple map with image overlay on top
            import sys
            sys.stderr.write(f"[DEBUG] Hillshade mode selected\n")
            sys.stderr.flush()

            # Create map with no base tiles initially
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)

            # Add hillshade overlay if available
            has_path = self.hillshade_path is not None
            has_meta = self.dem_metadata is not None
            file_exists = has_path and os.path.exists(self.hillshade_path)

            if has_path and has_meta and file_exists:
                try:
                    # Transform DEM bounds from EPSG:3794 to WGS84
                    from pyproj import Transformer
                    transformer = Transformer.from_crs("EPSG:3794", "EPSG:4326", always_xy=True)

                    x_min = float(self.dem_metadata['x_min'])
                    y_min = float(self.dem_metadata['y_min'])
                    x_max = float(self.dem_metadata['x_max'])
                    y_max = float(self.dem_metadata['y_max'])

                    lon_min, lat_min = transformer.transform(x_min, y_min)
                    lon_max, lat_max = transformer.transform(x_max, y_max)

                    # Use PIL to load and resize image
                    from PIL import Image
                    import io
                    import base64

                    img = Image.open(self.hillshade_path)
                    orig_size = img.size

                    # Resize for web (max 1200px for faster loading)
                    max_size = 1200
                    if max(img.size) > max_size:
                        ratio = max_size / max(img.size)
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)

                    # Keep as grayscale or convert to RGB (not RGBA - smaller file)
                    if img.mode == 'L':
                        img = img.convert('RGB')

                    # Convert to JPEG base64 (much smaller than PNG)
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    buffer.seek(0)
                    img_data = base64.b64encode(buffer.read()).decode('utf-8')

                    sys.stderr.write(f"[DEBUG] Hillshade: {orig_size} -> {img.size}, base64: {len(img_data)} bytes\n")
                    sys.stderr.flush()

                    # Add hillshade as image overlay FIRST (will be bottom layer)
                    img_overlay = folium.raster_layers.ImageOverlay(
                        image=f"data:image/jpeg;base64,{img_data}",
                        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                        opacity=1.0,
                        name='Hillshade',
                        overlay=True,
                        control=True,
                        show=True
                    )
                    img_overlay.add_to(m)

                    sys.stderr.write(f"[DEBUG] Hillshade overlay added\n")
                    sys.stderr.flush()

                except Exception as e:
                    sys.stderr.write(f"[ERROR] Hillshade failed: {e}\n")
                    sys.stderr.flush()
                    # Fallback to OSM
                    folium.TileLayer('OpenStreetMap').add_to(m)
            else:
                # No hillshade - use OSM
                folium.TileLayer('OpenStreetMap').add_to(m)

        # Add known caves if enabled and available
        if self.show_known_cb.isChecked() and self.known_caves is not None and self.data_extent is not None:
            min_x, min_y, max_x, max_y = self.data_extent
            # Filter caves within extent (with small buffer)
            buffer = 50  # 50m buffer
            filtered_caves = self.known_caves[
                (self.known_caves.geometry.x >= min_x - buffer) &
                (self.known_caves.geometry.x <= max_x + buffer) &
                (self.known_caves.geometry.y >= min_y - buffer) &
                (self.known_caves.geometry.y <= max_y + buffer)
            ]

            if len(filtered_caves) > 0:
                try:
                    caves_wgs = filtered_caves.to_crs(epsg=4326)
                    for idx, cave in caves_wgs.iterrows():
                        cave_lat = cave.geometry.y
                        cave_lon = cave.geometry.x
                        cave_name = cave.get('Ime', 'Unknown')
                        cave_type = cave.get('Tip_1', '')

                        popup_html = f"""
                        <div style="font-family: 'Segoe UI', Arial; min-width: 150px; padding: 5px;">
                            <h4 style="margin: 0 0 6px 0; color: #553c9a; border-bottom: 2px solid #805ad5; padding-bottom: 4px;">
                                {cave_name}
                            </h4>
                            <p style="margin: 0; font-size: 11px; color: #718096;">
                                <b>Type:</b> {cave_type}<br>
                                <i>Known cave (database)</i>
                            </p>
                        </div>
                        """

                        folium.CircleMarker(
                            location=[cave_lat, cave_lon],
                            radius=8,
                            color='#805ad5',
                            fill=True,
                            fill_color='#b794f4',
                            fill_opacity=0.7,
                            popup=folium.Popup(popup_html, max_width=200),
                            tooltip=cave_name
                        ).add_to(m)
                except Exception as e:
                    print(f"Error adding known caves: {e}")

        # Add markers for each detected cluster
        for idx, row in gdf_wgs.iterrows():
            cluster_id = int(row['cluster_id'])
            lat = row.geometry.y
            lon = row.geometry.x
            confidence = float(row['confidence_score'])
            max_depth = float(row['max_depth'])
            point_count = int(row['point_count'])
            shape_type = row['shape_type']

            # Store cluster data
            self.cluster_data[cluster_id] = {
                'centroid_x': float(row['centroid_x']),
                'centroid_y': float(row['centroid_y']),
                'confidence_score': confidence,
                'max_depth': max_depth,
                'point_count': point_count,
                'shape_type': shape_type,
                'lat': lat,
                'lon': lon
            }

            # Marker color based on confidence
            if confidence > 0.7:
                color = 'green'
            elif confidence > 0.5:
                color = 'orange'
            else:
                color = 'red'

            # Popup HTML with Show 3D button
            popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial; min-width: 180px; padding: 8px;">
                <h4 style="margin: 0 0 8px 0; color: #2d3748; border-bottom: 2px solid #4a6fa5; padding-bottom: 5px;">
                    Cluster #{cluster_id}
                </h4>
                <table style="font-size: 12px; width: 100%;">
                    <tr><td style="color: #718096;"><b>Points:</b></td><td style="text-align: right;">{point_count}</td></tr>
                    <tr><td style="color: #718096;"><b>Type:</b></td><td style="text-align: right;">{shape_type}</td></tr>
                    <tr><td style="color: #718096;"><b>Confidence:</b></td><td style="text-align: right;">{confidence:.2f}</td></tr>
                    <tr><td style="color: #718096;"><b>Max Depth:</b></td><td style="text-align: right;">{max_depth:.1f} m</td></tr>
                </table>
                <button onclick="show3D({cluster_id})" style="
                    margin-top: 8px; width: 100%; padding: 6px 12px;
                    background: #4a6fa5; color: white; border: none;
                    border-radius: 4px; cursor: pointer; font-weight: bold;
                    font-size: 11px;">
                    Show 3D View
                </button>
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=220),
                icon=folium.Icon(color=color, icon='info-sign', icon_color='white'),
                tooltip=f"#{cluster_id} - {confidence:.2f}"
            ).add_to(m)

        # Add locate control
        LocateControl(auto_start=False).add_to(m)

        # Get HTML content and add WebChannel script
        try:
            html_content = m.get_root().render()

            # Inject QWebChannel and show3D function
            webchannel_script = """
            <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
            <script>
                var bridge = null;
                new QWebChannel(qt.webChannelTransport, function(channel) {
                    bridge = channel.objects.bridge;
                });
                function show3D(clusterId) {
                    if (bridge) {
                        bridge.requestShow3D(clusterId);
                    } else {
                        console.log("Bridge not ready");
                    }
                }
            </script>
            """
            # Insert before </body>
            html_content = html_content.replace('</body>', webchannel_script + '</body>')

            self.current_html = html_content
            self.web_view.setHtml(html_content)
        except Exception as e:
            print(f"Map render error: {e}")
            self.show_placeholder()

    def highlight_cluster(self, cluster_id):
        """Pan map to a cluster by re-centering"""
        if cluster_id in self.cluster_data and self.gdf_clusters is not None:
            data = self.cluster_data[cluster_id]
            lat = data.get('lat')
            lon = data.get('lon')
            if lat and lon:
                # Try to pan using JavaScript - folium stores map as window.map_* variable
                js = f"""
                    (function() {{
                        for (var key in window) {{
                            if (key.startsWith('map_') && window[key] && window[key].setView) {{
                                window[key].setView([{lat}, {lon}], 17);
                                return;
                            }}
                        }}
                    }})();
                """
                self.web_view.page().runJavaScript(js)


class OptimizedTINViewer(QWidget):
    """Optimized 3D TIN viewer showing terrain with cave pits"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ground_points = None
        self.max_points = 3000
        self.grid_resolution = 40
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Create matplotlib figure with matching theme
        plt.style.use('dark_background')
        self.fig = plt.Figure(figsize=(6, 5), dpi=100, facecolor='#3d4a5c')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.style_axes()

        # Initial message
        self.ax.text2D(0.5, 0.5, "Izberi klaster za ogled",
                      transform=self.ax.transAxes, color='#90cdf4',
                      ha='center', va='center', fontsize=11)

        # Canvas
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, stretch=1)

        # Toolbar for pan/zoom/rotate
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Compact controls row
        controls = QHBoxLayout()
        controls.setSpacing(10)

        controls.addWidget(QLabel("Detail:"))
        self.res_slider = QSlider(Qt.Horizontal)
        self.res_slider.setRange(20, 80)
        self.res_slider.setValue(40)
        self.res_slider.setMaximumWidth(100)
        self.res_slider.setToolTip("Grid resolution for terrain rendering")
        self.res_slider.valueChanged.connect(self.on_resolution_change)
        controls.addWidget(self.res_slider)
        self.res_label = QLabel("40")
        self.res_label.setMinimumWidth(25)
        controls.addWidget(self.res_label)

        controls.addSpacing(15)

        self.show_terrain_cb = QCheckBox("Teren")
        self.show_terrain_cb.setChecked(True)
        self.show_terrain_cb.setToolTip("Pokaži teren")
        self.show_terrain_cb.stateChanged.connect(self.on_option_change)
        controls.addWidget(self.show_terrain_cb)

        self.show_pit_cb = QCheckBox("Oblika jame")
        self.show_pit_cb.setChecked(True)
        self.show_pit_cb.setToolTip("Vključi točke, da se pokaže oblika jame")
        self.show_pit_cb.stateChanged.connect(self.on_option_change)
        controls.addWidget(self.show_pit_cb)

        self.show_lines_cb = QCheckBox("Globina")
        self.show_lines_cb.setChecked(True)
        self.show_lines_cb.setToolTip("Pokaži linije, ki označujejo globino")
        self.show_lines_cb.stateChanged.connect(self.on_option_change)
        controls.addWidget(self.show_lines_cb)

        controls.addStretch()
        layout.addLayout(controls)

        # Store current render params for re-render
        self.current_params = None

    def style_axes(self):
        """Apply theme to 3D axes"""
        self.ax.set_facecolor('#3d4a5c')
        self.ax.tick_params(colors='#a0aec0', labelsize=7)
        self.ax.xaxis.label.set_color('#a0aec0')
        self.ax.yaxis.label.set_color('#a0aec0')
        self.ax.zaxis.label.set_color('#a0aec0')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#4a5568')
        self.ax.yaxis.pane.set_edgecolor('#4a5568')
        self.ax.zaxis.pane.set_edgecolor('#4a5568')
        self.ax.grid(True, alpha=0.2, color='#5a6578')

    def on_resolution_change(self, value):
        self.grid_resolution = value
        self.res_label.setText(str(value))
        if self.current_params:
            self.render_cluster(*self.current_params)

    def on_option_change(self):
        if self.current_params:
            self.render_cluster(*self.current_params)

    def set_ground_points(self, points):
        """Set ground points for rendering"""
        self.ground_points = points

    def render_cluster(self, cluster_id, centroid_x, centroid_y, low_points=None, radius=25.0):
        """Render terrain with cave pit visualization"""
        # Store params for re-render
        self.current_params = (cluster_id, centroid_x, centroid_y, low_points, radius)

        self.ax.clear()
        self.style_axes()

        if self.ground_points is None or len(self.ground_points) == 0:
            self.ax.text2D(0.5, 0.5, "No terrain data available",
                          transform=self.ax.transAxes, color='#fc8181',
                          ha='center', va='center')
            self.canvas.draw()
            return

        # Extract points within radius
        distances = np.sqrt((self.ground_points[:, 0] - centroid_x)**2 +
                           (self.ground_points[:, 1] - centroid_y)**2)
        nearby_mask = distances <= radius
        nearby = self.ground_points[nearby_mask]

        if len(nearby) < 4:
            self.ax.text2D(0.5, 0.5, f"Too few points ({len(nearby)})",
                          transform=self.ax.transAxes, color='#fc8181',
                          ha='center', va='center')
            self.canvas.draw()
            return

        try:
            # Subsample for performance
            if len(nearby) > self.max_points:
                indices = np.random.choice(len(nearby), self.max_points, replace=False)
                nearby_sampled = nearby[indices]
            else:
                nearby_sampled = nearby

            # Create regular grid
            x_min, x_max = nearby[:, 0].min(), nearby[:, 0].max()
            y_min, y_max = nearby[:, 1].min(), nearby[:, 1].max()

            xi = np.linspace(x_min, x_max, self.grid_resolution)
            yi = np.linspace(y_min, y_max, self.grid_resolution)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate terrain Z values
            zi_terrain = griddata(
                nearby_sampled[:, :2],
                nearby_sampled[:, 2],
                (xi, yi),
                method='linear'
            )

            # Get cave/low points within radius
            nearby_low = None
            if low_points is not None and len(low_points) > 0:
                low_distances = np.sqrt((low_points[:, 0] - centroid_x)**2 +
                                       (low_points[:, 1] - centroid_y)**2)
                nearby_low = low_points[low_distances <= radius]

            # Create combined surface showing pit
            if self.show_pit_cb.isChecked() and nearby_low is not None and len(nearby_low) > 0:
                all_points = np.vstack([nearby_sampled, nearby_low])
                zi_combined = griddata(
                    all_points[:, :2],
                    all_points[:, 2],
                    (xi, yi),
                    method='linear'
                )
                zi_display = zi_combined
            else:
                zi_display = zi_terrain

            # Plot terrain surface
            if self.show_terrain_cb.isChecked():
                self.ax.plot_surface(
                    xi, yi, zi_display,
                    cmap='terrain',
                    alpha=0.85,
                    edgecolor='none',
                    linewidth=0,
                    antialiased=True,
                    rcount=self.grid_resolution,
                    ccount=self.grid_resolution
                )

            # Plot cave candidate points
            if nearby_low is not None and len(nearby_low) > 0:
                self.ax.scatter(
                    nearby_low[:, 0], nearby_low[:, 1], nearby_low[:, 2],
                    c='#e53e3e', s=70, marker='o',
                    label=f'Cave points ({len(nearby_low)})',
                    depthshade=False, edgecolors='white', linewidths=0.8,
                    zorder=10
                )

                # Draw vertical lines from terrain to cave points
                if self.show_lines_cb.isChecked() and zi_terrain is not None:
                    for pt in nearby_low[:min(25, len(nearby_low))]:
                        ix = int((pt[0] - x_min) / (x_max - x_min + 0.001) * (self.grid_resolution - 1))
                        iy = int((pt[1] - y_min) / (y_max - y_min + 0.001) * (self.grid_resolution - 1))
                        ix = max(0, min(ix, self.grid_resolution - 1))
                        iy = max(0, min(iy, self.grid_resolution - 1))

                        terrain_z = zi_terrain[iy, ix]
                        if not np.isnan(terrain_z):
                            self.ax.plot(
                                [pt[0], pt[0]], [pt[1], pt[1]], [terrain_z, pt[2]],
                                color='#fc8181', linewidth=1.5, alpha=0.7
                            )

            # Mark centroid
            z_min = nearby[:, 2].min()
            if nearby_low is not None and len(nearby_low) > 0:
                z_min = min(z_min, nearby_low[:, 2].min())

            self.ax.scatter(
                [centroid_x], [centroid_y], [z_min],
                c='#f6e05e', s=150, marker='*',
                label='Center',
                depthshade=False, edgecolors='#744210', linewidths=1,
                zorder=11
            )

            # Labels
            self.ax.set_xlabel('X', fontsize=8, color='#a0aec0')
            self.ax.set_ylabel('Y', fontsize=8, color='#a0aec0')
            self.ax.set_zlabel('Z', fontsize=8, color='#a0aec0')
            self.ax.set_title(f'Cluster #{cluster_id}', color='#90cdf4', fontsize=11, pad=5)

            self.ax.legend(loc='upper left', fontsize=8, facecolor='#3d4a5c',
                          edgecolor='#4a5568', labelcolor='#e2e8f0')

            self.ax.set_box_aspect([1, 1, 0.5])

        except Exception as e:
            self.ax.text2D(0.5, 0.5, f"Error: {str(e)[:40]}",
                          transform=self.ax.transAxes, color='#fc8181',
                          ha='center', va='center', fontsize=9)

        self.canvas.draw()

    def clear(self):
        """Clear the view"""
        self.current_params = None
        self.ax.clear()
        self.style_axes()
        self.ax.text2D(0.5, 0.5, "Izberi klaster za 3D prikaz",
                      transform=self.ax.transAxes, color='#90cdf4',
                      ha='center', va='center', fontsize=11)
        self.canvas.draw()


class ClusterTable(QTableWidget):
    """Cluster list table with notes and potential marking"""

    cluster_selected = Signal(int, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cluster_data = {}
        self.notes_data = {}  # cluster_id -> notes string
        self.potential_data = {}  # cluster_id -> bool
        self.setup_ui()

    def setup_ui(self):
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels(['ID', 'Pts', 'Type', 'Conf', 'Depth', 'Pot.', 'Notes'])

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)

        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setSortingEnabled(True)

        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.cellChanged.connect(self.on_cell_changed)

    def populate(self, gdf_clusters):
        self.blockSignals(True)  # Prevent cellChanged during population
        self.setRowCount(0)
        self.cluster_data.clear()
        self.notes_data.clear()
        self.potential_data.clear()

        if gdf_clusters is None or len(gdf_clusters) == 0:
            self.blockSignals(False)
            return

        self.setSortingEnabled(False)

        for idx, row in gdf_clusters.iterrows():
            cluster_id = int(row['cluster_id'])
            point_count = int(row['point_count'])
            shape_type = row['shape_type']
            confidence = float(row['confidence_score'])
            max_depth = float(row['max_depth'])

            row_idx = self.rowCount()
            self.insertRow(row_idx)

            id_item = QTableWidgetItem()
            id_item.setData(Qt.DisplayRole, cluster_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row_idx, 0, id_item)

            pts_item = QTableWidgetItem()
            pts_item.setData(Qt.DisplayRole, point_count)
            pts_item.setFlags(pts_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row_idx, 1, pts_item)

            type_item = QTableWidgetItem(shape_type[:10])
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row_idx, 2, type_item)

            conf_item = QTableWidgetItem(f"{confidence:.2f}")
            conf_item.setFlags(conf_item.flags() & ~Qt.ItemIsEditable)
            if confidence > 0.7:
                conf_item.setForeground(QColor('#68d391'))
            elif confidence > 0.5:
                conf_item.setForeground(QColor('#f6ad55'))
            else:
                conf_item.setForeground(QColor('#fc8181'))
            self.setItem(row_idx, 3, conf_item)

            depth_item = QTableWidgetItem(f"{max_depth:.1f}m")
            depth_item.setFlags(depth_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row_idx, 4, depth_item)

            # Potential checkbox
            pot_item = QTableWidgetItem()
            pot_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            pot_item.setCheckState(Qt.Unchecked)
            self.setItem(row_idx, 5, pot_item)
            self.potential_data[cluster_id] = False

            # Notes (editable)
            notes_item = QTableWidgetItem("")
            notes_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.setItem(row_idx, 6, notes_item)
            self.notes_data[cluster_id] = ""

            self.cluster_data[cluster_id] = {
                'centroid_x': float(row['centroid_x']),
                'centroid_y': float(row['centroid_y']),
                'confidence_score': confidence,
                'max_depth': max_depth,
                'point_count': point_count,
                'shape_type': shape_type
            }

        self.blockSignals(False)
        self.setSortingEnabled(True)
        self.sortByColumn(3, Qt.DescendingOrder)

    def on_selection_changed(self):
        rows = self.selectedItems()
        if rows:
            row_idx = rows[0].row()
            cluster_id = int(self.item(row_idx, 0).data(Qt.DisplayRole))
            if cluster_id in self.cluster_data:
                self.cluster_selected.emit(cluster_id, self.cluster_data[cluster_id])

    def on_cell_changed(self, row, col):
        """Handle cell edits for notes and potential checkbox"""
        if col == 5:  # Potential checkbox
            cluster_id = int(self.item(row, 0).data(Qt.DisplayRole))
            is_checked = self.item(row, 5).checkState() == Qt.Checked
            self.potential_data[cluster_id] = is_checked
        elif col == 6:  # Notes
            cluster_id = int(self.item(row, 0).data(Qt.DisplayRole))
            notes_text = self.item(row, 6).text()
            self.notes_data[cluster_id] = notes_text

    def select_cluster(self, cluster_id):
        """Programmatically select a cluster"""
        for row in range(self.rowCount()):
            if int(self.item(row, 0).data(Qt.DisplayRole)) == cluster_id:
                self.selectRow(row)
                self.scrollToItem(self.item(row, 0))
                break

    def get_export_data(self, potential_only=False):
        """Get cluster data with notes and potential for export"""
        export_list = []
        for cluster_id, data in self.cluster_data.items():
            is_potential = self.potential_data.get(cluster_id, False)
            if potential_only and not is_potential:
                continue
            export_list.append({
                'cluster_id': cluster_id,
                'centroid_x': data['centroid_x'],
                'centroid_y': data['centroid_y'],
                'confidence_score': data['confidence_score'],
                'max_depth': data['max_depth'],
                'point_count': data['point_count'],
                'shape_type': data['shape_type'],
                'potential': is_potential,
                'notes': self.notes_data.get(cluster_id, "")
            })
        return export_list

    def get_potential_count(self):
        """Count clusters marked as potential"""
        return sum(1 for v in self.potential_data.values() if v)


class CaveDetectionApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.results = ProcessingResults()
        self.worker = None
        self.input_files = []

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        self.setWindowTitle("Chekich 0.8 - Orodje za iskanje vhodov v jame")
        self.setMinimumSize(1400, 850)
        self.setStyleSheet(SOFT_STYLE)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # === LEFT PANEL ===
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # === RIGHT PANEL (Tabs) ===
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([340, 1060])

        self.statusBar().showMessage("Pripravlen - Izberi  LAZ/LAS datoteko")

    def create_left_panel(self):
        """Create left panel with controls and cluster list"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title = QLabel("Checkich v0.8")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Detekcija vhodov v jame")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #a0aec0; font-size: 9pt; margin-bottom: 5px;")
        layout.addWidget(subtitle)

        # File selection
        file_group = QGroupBox("Input")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(6)

        self.file_entry = QLineEdit()
        self.file_entry.setPlaceholderText("Izberi LAZ/LAS files...")
        self.file_entry.setReadOnly(True)
        file_layout.addWidget(self.file_entry)

        btn_row = QHBoxLayout()
        browse_btn = QPushButton("Brskaj")
        browse_btn.clicked.connect(self.browse_files)
        btn_row.addWidget(browse_btn)

        output_btn = QPushButton("Output...")
        output_btn.clicked.connect(self.browse_output)
        output_btn.setMaximumWidth(80)
        btn_row.addWidget(output_btn)
        file_layout.addLayout(btn_row)

        self.output_entry = QLineEdit(str(Path.cwd()))
        self.output_entry.setPlaceholderText("Output folder")
        file_layout.addWidget(self.output_entry)

        layout.addWidget(file_group)

        # Settings
        settings_group = QGroupBox("Natavitve detekcije")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setSpacing(8)

        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(["konzervativno", "srednje", "občutljivo"])
        self.sensitivity_combo.setCurrentIndex(1)
        self.sensitivity_combo.currentIndexChanged.connect(self.on_sensitivity_change)
        settings_layout.addRow("Občutljivost:", self.sensitivity_combo)

        params_row = QHBoxLayout()
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(1.0, 20.0)
        self.radius_spin.setValue(5.0)
        self.radius_spin.setSuffix(" m")
        params_row.addWidget(QLabel("Radius:"))
        params_row.addWidget(self.radius_spin)

        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.5, 10.0)
        self.depth_spin.setValue(3.0)
        self.depth_spin.setSuffix(" m")
        params_row.addWidget(QLabel("Globina:"))
        params_row.addWidget(self.depth_spin)
        settings_layout.addRow(params_row)

        self.include_singles = QCheckBox("Vključi osamele točke")
        settings_layout.addRow(self.include_singles)

        layout.addWidget(settings_group)

        # Start button
        self.start_btn = QPushButton("ZAČNI DETEKCIJO")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)

        # Progress
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        progress_row.addWidget(self.progress_bar)
        self.status_label = QLabel("Pripravljen")
        self.status_label.setMinimumWidth(70)
        progress_row.addWidget(self.status_label)
        layout.addLayout(progress_row)

        # Cluster list
        cluster_group = QGroupBox("Najdene gruče")
        cluster_layout = QVBoxLayout(cluster_group)

        self.cluster_table = ClusterTable()
        self.cluster_table.cluster_selected.connect(self.on_cluster_selected)
        cluster_layout.addWidget(self.cluster_table)

        self.summary_label = QLabel("Ni še rezultatov")
        self.summary_label.setStyleSheet("color: #a0aec0;")
        cluster_layout.addWidget(self.summary_label)

        # Export buttons
        export_row = QHBoxLayout()
        self.export_all_btn = QPushButton("Izvozi vse")
        self.export_all_btn.setToolTip("Izvozi vse klastre z opombami in polji")
        self.export_all_btn.clicked.connect(self.export_all_clusters)
        self.export_all_btn.setEnabled(False)
        export_row.addWidget(self.export_all_btn)

        self.export_potential_btn = QPushButton("Izvozi potencialne")
        self.export_potential_btn.setToolTip("Izvozi samo potencialne")
        self.export_potential_btn.clicked.connect(self.export_potential_clusters)
        self.export_potential_btn.setEnabled(False)
        export_row.addWidget(self.export_potential_btn)
        cluster_layout.addLayout(export_row)

        layout.addWidget(cluster_group, stretch=1)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        return panel

    def create_right_panel(self):
        """Create right panel with Map/3D tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()

        # Map tab
        self.map_widget = InteractiveMapWidget()
        self.map_widget.show_3d_requested.connect(self.on_show_3d_from_map)
        self.tab_widget.addTab(self.map_widget, "  Map View  ")

        # 3D tab
        self.tin_viewer = OptimizedTINViewer()
        self.tab_widget.addTab(self.tin_viewer, "  3D teren  ")

        layout.addWidget(self.tab_widget)

        return panel

    def on_show_3d_from_map(self, cluster_id):
        """Handle Show 3D button click from map popup"""
        # Select the cluster in the table
        self.cluster_table.select_cluster(cluster_id)
        # Switch to 3D tab
        self.tab_widget.setCurrentIndex(1)

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Izberi LIDAR datoteko",
            str(Path.cwd()), "LAZ/LAS datoteke (*.laz *.las);;All Files (*.*)"
        )
        if files:
            self.input_files = files
            names = [os.path.basename(f) for f in files]
            self.file_entry.setText("; ".join(names))
            self.log(f"Selected {len(files)} file(s)")

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "izberi izhodno mapo", self.output_entry.text())
        if folder:
            self.output_entry.setText(folder)

    def on_sensitivity_change(self, index):
        presets = [(3.0, 4.0), (5.0, 3.0), (7.0, 2.0)]
        radius, depth = presets[index]
        self.radius_spin.setValue(radius)
        self.depth_spin.setValue(depth)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def start_processing(self):
        if not self.input_files:
            QMessageBox.warning(self, "Ni vhodnih podatkov", "Najprej izberi vhodne podatke")
            return

        self.start_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Procesiranje...")
        self.status_label.setStyleSheet("color: #90cdf4;")
        self.log("Začenjam detekcijo...")

        self.cluster_table.setRowCount(0)
        self.tin_viewer.clear()
        self.map_widget.show_placeholder()

        config = DetectionConfig(
            radius=self.radius_spin.value(),
            depth_threshold=self.depth_spin.value(),
            min_ground_points=10,
            cluster_eps=3.0,
            cluster_min_samples=2,
            strict_depth_filter=self.depth_spin.value(),
            include_single_points=self.include_singles.isChecked()
        )

        input_file = self.input_files[0]
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_base = os.path.join(self.output_entry.text(), base_name)

        self.worker = DetectionWorker(input_file, output_base, config)
        self.worker.signals.log.connect(self.log)
        self.worker.signals.finished.connect(self.on_processing_finished)
        self.worker.signals.error.connect(self.on_processing_error)
        self.worker.start()

    def on_processing_finished(self, success, data):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else 0)
        self.start_btn.setEnabled(True)

        if success and data:
            self.status_label.setText("Končano!")
            self.status_label.setStyleSheet("color: #68d391;")
            self.log("Detekcija končana!")

            self.results.gdf_points = data.get('gdf_points')
            self.results.gdf_clusters = data.get('gdf_clusters')
            self.results.ground_points = data.get('ground_points')
            self.results.potential_cave_points = data.get('potential_cave_points')
            self.results.dem = data.get('dem')
            self.results.dem_metadata = data.get('dem_metadata')
            self.results.hillshade_path = data.get('hillshade_path')

            # Set data extent and hillshade for map widget
            if self.results.dem_metadata is not None:
                meta = self.results.dem_metadata
                self.map_widget.set_data_extent(meta['x_min'], meta['y_min'], meta['x_max'], meta['y_max'])
                self.log(f"DEM extent: X[{meta['x_min']:.0f}-{meta['x_max']:.0f}] Y[{meta['y_min']:.0f}-{meta['y_max']:.0f}]")

            # Set hillshade for basemap option
            if self.results.hillshade_path and self.results.dem_metadata:
                self.map_widget.set_hillshade(self.results.hillshade_path, self.results.dem_metadata)
                self.log(f"Hillshade available: {os.path.basename(self.results.hillshade_path)}")

            # Update UI
            self.cluster_table.populate(self.results.gdf_clusters)
            self.tin_viewer.set_ground_points(self.results.ground_points)
            self.map_widget.update_map(self.results.gdf_clusters, self.results.gdf_points)

            total = 0
            if self.results.gdf_clusters is not None:
                total = len(self.results.gdf_clusters)
                high_conf = len(self.results.gdf_clusters[
                    self.results.gdf_clusters['confidence_score'] > 0.7
                ])
                self.summary_label.setText(f"{total} clusters ({high_conf} high confidence)")
                self.summary_label.setStyleSheet("color: #68d391;")

            # Enable export buttons
            has_clusters = total > 0
            self.export_all_btn.setEnabled(has_clusters)
            self.export_potential_btn.setEnabled(has_clusters)

            self.statusBar().showMessage(f"Detection končano - {total} najdenih klastrov")
        else:
            self.status_label.setText("Neuspešno")
            self.status_label.setStyleSheet("color: #fc8181;")
            self.log("Detekcija neuspešna - preveri vhodne podatke")

    def on_processing_error(self, error_msg):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(True)
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("color: #fc8181;")
        self.log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Processing failed:\n{error_msg}")

    def on_cluster_selected(self, cluster_id, cluster_data):
        """Handle cluster selection from table"""
        self.log(f"Selected cluster #{cluster_id}")

        centroid_x = cluster_data['centroid_x']
        centroid_y = cluster_data['centroid_y']

        low_points = None
        if self.results.gdf_points is not None:
            cluster_pts = self.results.gdf_points[
                self.results.gdf_points['cluster'] == cluster_id
            ]
            if len(cluster_pts) > 0:
                low_points = np.column_stack([
                    cluster_pts['X'].values,
                    cluster_pts['Y'].values,
                    cluster_pts['Z'].values
                ])

        self.tin_viewer.render_cluster(cluster_id, centroid_x, centroid_y, low_points)
        self.map_widget.highlight_cluster(cluster_id)

    def export_all_clusters(self):
        """Export all clusters with notes and potential fields to shapefile"""
        self._export_clusters(potential_only=False)

    def export_potential_clusters(self):
        """Export only potential clusters to shapefile"""
        self._export_clusters(potential_only=True)

    def _export_clusters(self, potential_only=False):
        """Export clusters to shapefile with notes and potential fields"""
        export_data = self.cluster_table.get_export_data(potential_only=potential_only)

        if not export_data:
            if potential_only:
                QMessageBox.warning(self, "No Data",
                    "No clusters marked as potential. Check the 'Pot.' checkbox for clusters you want to export.")
            else:
                QMessageBox.warning(self, "No Data", "No clusters to export.")
            return

        # Ask for save location
        suffix = "_potential" if potential_only else "_all"
        default_name = f"clusters{suffix}.shp"
        output_folder = self.output_entry.text() or str(Path.cwd())

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Shapefile",
            os.path.join(output_folder, default_name),
            "Shapefile (*.shp)"
        )

        if not file_path:
            return

        try:
            from shapely.geometry import Point

            # Create GeoDataFrame
            geometries = [Point(d['centroid_x'], d['centroid_y']) for d in export_data]

            gdf = gpd.GeoDataFrame({
                'cluster_id': [d['cluster_id'] for d in export_data],
                'point_cnt': [d['point_count'] for d in export_data],
                'shape_type': [d['shape_type'][:10] for d in export_data],
                'confidence': [d['confidence_score'] for d in export_data],
                'max_depth': [d['max_depth'] for d in export_data],
                'potential': [1 if d['potential'] else 0 for d in export_data],
                'notes': [d['notes'][:254] for d in export_data],  # Shapefile limit
                'geometry': geometries
            }, crs="EPSG:3794")

            gdf.to_file(file_path)

            count = len(export_data)
            potential_count = sum(1 for d in export_data if d['potential'])
            self.log(f"Exported {count} clusters to {file_path}")
            if potential_only:
                QMessageBox.information(self, "Export Complete",
                    f"Exported {count} potential clusters to:\n{file_path}")
            else:
                QMessageBox.information(self, "Export Complete",
                    f"Exported {count} clusters ({potential_count} marked as potential) to:\n{file_path}")

        except Exception as e:
            self.log(f"Export error: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")

    def save_settings(self):
        data = {
            "output_folder": self.output_entry.text(),
            "sensitivity": self.sensitivity_combo.currentIndex(),
            "radius": self.radius_spin.value(),
            "depth": self.depth_spin.value(),
            "include_singles": self.include_singles.isChecked()
        }
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except:
            pass

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                self.output_entry.setText(data.get("output_folder", str(Path.cwd())))
                self.sensitivity_combo.setCurrentIndex(data.get("sensitivity", 1))
                self.radius_spin.setValue(data.get("radius", 5.0))
                self.depth_spin.setValue(data.get("depth", 3.0))
                self.include_singles.setChecked(data.get("include_singles", False))
            except:
                pass

    def closeEvent(self, event):
        self.save_settings()
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = CaveDetectionApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
