"""
Enhanced layout visualization utilities inspired by the “Fixed FBS Architectural
Layout Generator” workflow provided by the user. This module focuses on:

1. Compact room placement heuristics (optional refinement step)
2. Strict adjacency detection derived from actual geometry, not only desired
   adjacency metadata
3. Rich SVG floor-plan generation with compass, legends, and room metadata
4. Diagnostic adjacency graphs rendered via Matplotlib/NetworkX that classify
   edges (critical/preferred/spatial) and ensure graph connectivity

The goal is to provide higher-quality visuals and clearer adjacency insights
than the legacy SVG snapshots.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon


@dataclass
class VisualConfig:
    """Styling options used by both SVG and Matplotlib outputs."""

    wall_color: str = "#2C3E50"
    wall_width: float = 0.2
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    title_font_size: int = 18
    margin: float = 50.0
    scale: float = 30.0  # pixels per meter
    show_grid: bool = True
    grid_color: str = "#ECF0F1"
    grid_spacing: float = 1.0  # meters
    show_dimensions: bool = True
    dimension_color: str = "#E74C3C"
    show_directions: bool = True
    compass_size: float = 60.0
    direction_arrow_color: str = "#F39C12"
    sun_color: str = "#F1C40F"
    wind_color: str = "#3498DB"
    node_size: float = 900.0
    edge_width: float = 2.0
    output_root: Path = field(default_factory=lambda: Path("visual_outputs"))

    room_colors: Dict[str, str] = field(default_factory=lambda: {
        "living_room": "#9B59B6",
        "dining_room": "#F39C12",
        "dining_hall": "#F39C12",
        "kitchen": "#27AE60",
        "bedroom": "#3498DB",
        "bathroom": "#E67E22",
        "study": "#E74C3C",
        "office": "#E74C3C",
        "corridor": "#BDC3C7",
        "hallway": "#BDC3C7",
        "utility": "#16A085",
        "storage": "#8E44AD",
        "garage": "#95A5A6",
        "default": "#34495E",
    })


class LayoutRoomAdapter:
    """Utility that converts our Layout / Room structures into dictionaries expected
    by the enhanced visualization pipeline."""

    def __init__(self, layout) -> None:
        self.layout = layout

    def extract_rooms(self) -> List[Dict[str, Any]]:
        rooms = []
        if not getattr(self.layout, "room_polygons", None):
            return rooms

        for room_id, polygon in self.layout.room_polygons.items():
            if not isinstance(polygon, Polygon):
                continue

            bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            area = polygon.area if hasattr(polygon, "area") else width * height
            room = self.layout.rooms.get(room_id)
            room_type = getattr(room, "room_type", "default") if room else "default"

            rooms.append({
                "room_id": room_id,
                "room_type": room_type,
                "area": area,
                "x": bounds[0],
                "y": bounds[1],
                "width": width,
                "height": height,
                "natural_light_access": True,
                "adjacencies": getattr(room, "required_adjacencies", []) or [],
            })
        return rooms


class CompactRoomPlacer:
    """Heuristic placement refinements inspired by the user's reference code.
    We only use it opportunistically: if the layout already has coordinates we
    keep them, otherwise we fall back to compact placement."""

    importance_scores = {
        "living_room": 15,
        "kitchen": 14,
        "hallway": 13,
        "corridor": 13,
        "dining_room": 12,
        "dining_hall": 12,
        "bedroom": 10,
        "bathroom": 8,
        "study": 7,
        "office": 7,
        "utility": 5,
        "storage": 4,
        "garage": 3,
    }

    @classmethod
    def prioritize(cls, rooms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def score(room: Dict[str, Any]) -> float:
            base = cls.importance_scores.get(room["room_type"], 5)
            adjacency_boost = len(room.get("adjacencies", [])) * 2
            area_boost = min(room.get("area", 0) / 20.0, 3)
            return base + adjacency_boost + area_boost

        return sorted(rooms, key=score, reverse=True)

    @classmethod
    def place(cls, rooms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rooms:
            return rooms

        prioritized = cls.prioritize(rooms)
        placed = []

        first = prioritized[0].copy()
        first["x"], first["y"] = 0.0, 0.0
        placed.append(first)

        for room in prioritized[1:]:
            candidate = cls._find_slot(room, placed)
            room_copy = room.copy()
            room_copy["x"], room_copy["y"] = candidate
            placed.append(room_copy)

        return placed

    @classmethod
    def _find_slot(cls, room: Dict[str, Any], placed: List[Dict[str, Any]]) -> Tuple[float, float]:
        spacing = 0.5
        best_pos = None
        best_score = float("-inf")

        for existing in placed:
            candidate_positions = [
                (existing["x"] + existing["width"] + spacing, existing["y"]),
                (existing["x"] - room["width"] - spacing, existing["y"]),
                (existing["x"], existing["y"] + existing["height"] + spacing),
                (existing["x"], existing["y"] - room["height"] - spacing),
            ]

            for pos in candidate_positions:
                if cls._overlaps(room, pos, placed):
                    continue
                score = cls._evaluate(room, pos, existing)
                if score > best_score:
                    best_score = score
                    best_pos = pos

        if best_pos is None:
            max_x = max(r["x"] + r["width"] for r in placed)
            best_pos = (max_x + spacing, placed[0]["y"])

        return best_pos

    @staticmethod
    def _evaluate(room: Dict[str, Any], pos: Tuple[float, float], anchor: Dict[str, Any]) -> float:
        room_center = (pos[0] + room["width"] / 2, pos[1] + room["height"] / 2)
        anchor_center = (anchor["x"] + anchor["width"] / 2, anchor["y"] + anchor["height"] / 2)
        distance = math.dist(room_center, anchor_center)
        return 100.0 / (distance + 1.0)

    @staticmethod
    def _overlaps(room: Dict[str, Any], pos: Tuple[float, float], placed: List[Dict[str, Any]]) -> bool:
        x, y = pos
        tolerance = 0.1
        rect1 = (x - tolerance, y - tolerance,
                 x + room["width"] + tolerance, y + room["height"] + tolerance)

        for other in placed:
            rect2 = (other["x"] - tolerance, other["y"] - tolerance,
                     other["x"] + other["width"] + tolerance, other["y"] + other["height"] + tolerance)
            if not (rect1[2] <= rect2[0] or rect2[2] <= rect1[0] or
                    rect1[3] <= rect2[1] or rect2[3] <= rect1[1]):
                return True
        return False


class EnhancedLayoutVisualizer:
    """Main entry point that orchestrates SVG + adjacency outputs."""

    def __init__(self, config: Optional[VisualConfig] = None) -> None:
        self.config = config or VisualConfig()
        self.config.output_root.mkdir(parents=True, exist_ok=True)

    def render(self, layout, project_name: str, node_id: str) -> Dict[str, str]:
        room_adapter = LayoutRoomAdapter(layout)
        rooms = room_adapter.extract_rooms()
        if not rooms:
            return {}

        bounds = self._calculate_bounds(rooms)
        margin = self.config.margin
        width = (bounds["max_x"] - bounds["min_x"] + 2 * margin) * self.config.scale
        height = (bounds["max_y"] - bounds["min_y"] + 2 * margin) * self.config.scale

        svg = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{width}",
            "height": f"{height}",
            "viewBox": f"0 0 {width} {height}",
        })

        self._add_styles(svg)
        if self.config.show_grid:
            self._add_grid(svg, bounds, margin)
        if self.config.show_directions:
            self._add_compass(svg, width, height)

        for room in rooms:
            self._add_room(svg, room, bounds, margin)

        title = ET.SubElement(svg, "text", {
            "x": "20",
            "y": "35",
            "class": "title-text",
        })
        title.text = f"{project_name.replace('_', ' ').title()} – Layout for {node_id[:8]}"

        total_area = sum(r.get("area", 0) for r in rooms)
        legend = ET.SubElement(svg, "text", {
            "x": "20",
            "y": "60",
            "class": "subtitle-text",
        })
        legend.text = f"Total area: {total_area:.1f} m²   Rooms: {len(rooms)}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        svg_path = self.config.output_root / f"{project_name}_{node_id[:8]}_layout_{timestamp}.svg"
        tree = ET.ElementTree(svg)
        ET.indent(tree, space="  ")
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)

        # Generate adjacency graph
        layout_data = {"rooms": rooms, "total_area": total_area}
        adjacency_path = self._generate_adjacency_graph(layout_data, project_name, node_id)

        return {"svg_path": str(svg_path), "adjacency_path": str(adjacency_path)}

    def _calculate_bounds(self, rooms: List[Dict[str, Any]]) -> Dict[str, float]:
        min_x = min(room["x"] for room in rooms)
        min_y = min(room["y"] for room in rooms)
        max_x = max(room["x"] + room["width"] for room in rooms)
        max_y = max(room["y"] + room["height"] for room in rooms)
        return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}

    def _add_styles(self, svg: ET.Element) -> None:
        style = ET.SubElement(svg, "style")
        style.text = f"""
        .room {{
            stroke: {self.config.wall_color};
            stroke-width: {self.config.wall_width};
            opacity: 0.95;
        }}
        .room-label {{
            font-family: {self.config.font_family};
            font-size: {self.config.font_size}px;
            font-weight: bold;
            fill: #ffffff;
            text-anchor: middle;
        }}
        .title-text {{
            font-family: {self.config.font_family};
            font-size: {self.config.title_font_size}px;
            font-weight: bold;
            fill: {self.config.wall_color};
        }}
        .subtitle-text {{
            font-family: {self.config.font_family};
            font-size: {self.config.font_size}px;
            fill: {self.config.wall_color};
        }}
        .grid-line {{
            stroke: {self.config.grid_color};
            stroke-width: 0.5;
            opacity: 0.4;
        }}
        """

    def _add_grid(self, svg: ET.Element, bounds: Dict[str, float], margin: float) -> None:
        width = (bounds["max_x"] - bounds["min_x"] + 2 * margin) * self.config.scale
        height = (bounds["max_y"] - bounds["min_y"] + 2 * margin) * self.config.scale
        group = ET.SubElement(svg, "g", {"class": "grid"})

        spacing = self.config.grid_spacing
        x = bounds["min_x"] - margin
        while x <= bounds["max_x"] + margin:
            line = ET.SubElement(group, "line", {
                "x1": f"{(x - bounds['min_x'] + margin) * self.config.scale}",
                "y1": "0",
                "x2": f"{(x - bounds['min_x'] + margin) * self.config.scale}",
                "y2": f"{height}",
                "class": "grid-line",
            })
            x += spacing

        y = bounds["min_y"] - margin
        while y <= bounds["max_y"] + margin:
            line = ET.SubElement(group, "line", {
                "x1": "0",
                "y1": f"{(y - bounds['min_y'] + margin) * self.config.scale}",
                "x2": f"{width}",
                "y2": f"{(y - bounds['min_y'] + margin) * self.config.scale}",
                "class": "grid-line",
            })
            y += spacing

    def _add_compass(self, svg: ET.Element, width: float, height: float) -> None:
        group = ET.SubElement(svg, "g", {
            "transform": f"translate({width - self.config.compass_size - 20}, 20)",
        })
        ET.SubElement(group, "circle", {
            "cx": f"{self.config.compass_size / 2}",
            "cy": f"{self.config.compass_size / 2}",
            "r": f"{self.config.compass_size / 2}",
            "fill": "#ffffff",
            "stroke": self.config.wall_color,
            "stroke-width": "2",
        })
        arrow = ET.SubElement(group, "polygon", {
            "points": f"{self.config.compass_size/2},{self.config.compass_size/2-15} "
                      f"{self.config.compass_size/2-5},{self.config.compass_size/2-5} "
                      f"{self.config.compass_size/2+5},{self.config.compass_size/2-5}",
            "fill": self.config.direction_arrow_color,
        })
        arrow.set("opacity", "0.9")

    def _add_room(self, svg: ET.Element, room: Dict[str, Any],
                  bounds: Dict[str, float], margin: float) -> None:
        x = (room["x"] - bounds["min_x"] + margin) * self.config.scale
        y = (room["y"] - bounds["min_y"] + margin) * self.config.scale
        width = room["width"] * self.config.scale
        height = room["height"] * self.config.scale

        fill = self.config.room_colors.get(room["room_type"], self.config.room_colors["default"])
        rect = ET.SubElement(svg, "rect", {
            "x": f"{x}",
            "y": f"{y}",
            "width": f"{width}",
            "height": f"{height}",
            "fill": fill,
            "class": "room",
        })
        label = ET.SubElement(svg, "text", {
            "x": f"{x + width / 2}",
            "y": f"{y + height / 2}",
            "class": "room-label",
        })
        label.text = room["room_type"].replace("_", " ").title()

    # ----------------------------------------------------------- Adjacency Graph
    def _generate_adjacency_graph(self, layout_data: Dict[str, Any],
                                  project_name: str, node_id: str) -> str:
        rooms = layout_data["rooms"]
        graph = nx.Graph()

        for room in rooms:
            graph.add_node(room["room_id"],
                           room_type=room["room_type"],
                           area=room["area"],
                           x=room["x"],
                           y=room["y"],
                           width=room["width"],
                           height=room["height"])

        self._add_adjacency_edges(graph, rooms)
        self._ensure_connectivity(graph, rooms)

        fig, (ax_layout, ax_graph) = plt.subplots(1, 2, figsize=(18, 9))
        fig.suptitle(f"{project_name.replace('_', ' ').title()} – Connectivity ({node_id[:8]})",
                     fontsize=16, fontweight="bold")

        self._plot_spatial(ax_layout, graph)
        self._plot_graph(ax_graph, graph)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = self.config.output_root / f"{project_name}_{node_id}_adjacency_{timestamp}.png"
        fig.tight_layout()
        fig.savefig(graph_path, dpi=200)
        plt.close(fig)
        return str(graph_path)

    def _add_adjacency_edges(self, graph: nx.Graph, rooms: List[Dict[str, Any]]) -> None:
        for i, room_a in enumerate(rooms):
            for j in range(i + 1, len(rooms)):
                room_b = rooms[j]
                if self._rooms_adjacent(room_a, room_b):
                    graph.add_edge(room_a["room_id"], room_b["room_id"],
                                   edge_type=self._edge_type(room_a, room_b))

        if graph.number_of_edges() == 0:
            self._add_distance_edges(graph, rooms)

    def _edge_type(self, room_a: Dict[str, Any], room_b: Dict[str, Any]) -> str:
        critical = {"kitchen": {"dining_room", "living_room"},
                    "bedroom": {"bathroom", "corridor", "hallway"}}
        if room_b["room_type"] in critical.get(room_a["room_type"], set()):
            return "critical"
        if room_a["room_type"] in critical.get(room_b["room_type"], set()):
            return "critical"
        return "spatial"

    def _add_distance_edges(self, graph: nx.Graph, rooms: List[Dict[str, Any]]) -> None:
        pairs = []
        for i, room_a in enumerate(rooms):
            for j in range(i + 1, len(rooms)):
                room_b = rooms[j]
                dist = math.dist(
                    (room_a["x"] + room_a["width"] / 2, room_a["y"] + room_a["height"] / 2),
                    (room_b["x"] + room_b["width"] / 2, room_b["y"] + room_b["height"] / 2),
                )
                pairs.append((dist, room_a, room_b))
        pairs.sort(key=lambda x: x[0])

        for dist, room_a, room_b in pairs[:max(1, len(rooms) - 1)]:
            graph.add_edge(room_a["room_id"], room_b["room_id"], edge_type="proximity")

    def _ensure_connectivity(self, graph: nx.Graph, rooms: List[Dict[str, Any]]) -> None:
        if nx.is_connected(graph):
            return
        components = list(nx.connected_components(graph))
        main = components[0]
        for comp in components[1:]:
            node_a = next(iter(comp))
            node_b = next(iter(main))
            graph.add_edge(node_a, node_b, edge_type="bridge")
            main.update(comp)

    def _rooms_adjacent(self, room1: Dict[str, Any], room2: Dict[str, Any]) -> bool:
        tol = 0.2
        left1, right1 = room1["x"], room1["x"] + room1["width"]
        top1, bottom1 = room1["y"], room1["y"] + room1["height"]
        left2, right2 = room2["x"], room2["x"] + room2["width"]
        top2, bottom2 = room2["y"], room2["y"] + room2["height"]

        horizontal = abs(right1 - left2) <= tol or abs(right2 - left1) <= tol
        vertical = abs(bottom1 - top2) <= tol or abs(bottom2 - top1) <= tol
        if horizontal:
            overlap = min(bottom1, bottom2) - max(top1, top2)
            return overlap >= min(room1["height"], room2["height"]) * 0.3
        if vertical:
            overlap = min(right1, right2) - max(left1, left2)
            return overlap >= min(room1["width"], room2["width"]) * 0.3
        return False

    def _plot_spatial(self, ax, graph: nx.Graph) -> None:
        ax.set_title("Spatial Layout")
        for node, data in graph.nodes(data=True):
            rect = plt.Rectangle(
                (data["x"], data["y"]),
                data["width"],
                data["height"],
                edgecolor="black",
                facecolor=self.config.room_colors.get(data["room_type"], "#999999"),
                alpha=0.7,
            )
            ax.add_patch(rect)
            ax.text(
                data["x"] + data["width"] / 2,
                data["y"] + data["height"] / 2,
                data["room_type"].replace("_", " ").title(),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        for u, v, edge in graph.edges(data=True):
            xa = graph.nodes[u]["x"] + graph.nodes[u]["width"] / 2
            ya = graph.nodes[u]["y"] + graph.nodes[u]["height"] / 2
            xb = graph.nodes[v]["x"] + graph.nodes[v]["width"] / 2
            yb = graph.nodes[v]["y"] + graph.nodes[v]["height"] / 2
            ax.plot([xa, xb], [ya, yb], color="#2C3E50", linewidth=1.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    def _plot_graph(self, ax, graph: nx.Graph) -> None:
        ax.set_title("Adjacency Graph")
        pos = nx.spring_layout(graph, seed=42)
        node_colors = [
            self.config.room_colors.get(data["room_type"], "#999999")
            for _, data in graph.nodes(data=True)
        ]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                               node_size=self.config.node_size, ax=ax, edgecolors="black")
        edge_styles = {
            "critical": ("#27AE60", 3.5, "solid"),
            "spatial": ("#3498DB", 2.5, "solid"),
            "proximity": ("#95A5A6", 1.5, "dotted"),
            "bridge": ("#E74C3C", 2.0, "dashed"),
        }
        for edge_type, (color, width, style) in edge_styles.items():
            edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == edge_type]
            if edges:
                nx.draw_networkx_edges(graph, pos, edgelist=edges, ax=ax,
                                       edge_color=color, width=width, style=style, alpha=0.8,
                                       label=f"{edge_type.title()} ({len(edges)})")
        labels = {node: data["room_type"].replace("_", " ").title() for node, data in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels, ax=ax, font_size=9, font_weight="bold")
        ax.axis("off")
        ax.legend(loc="lower left")


