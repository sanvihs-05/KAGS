"""
Validation Framework for FBSL-KAGS Layout Design Results
This script validates design prototypes and strategies across multiple dimensions:
- Data integrity and schema compliance
- Functional requirements satisfaction
- Behavioral performance targets
- Structural feasibility
- Layout metrics and spatial constraints
- Score consistency and calculation accuracy
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math


class LayoutValidator:
    """Validates layout design results across multiple criteria"""
    
    def __init__(self):
        self.validation_results = {}
        self.warnings = []
        self.errors = []
        
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Main validation entry point for a single JSON file"""
        print(f"\n{'='*80}")
        print(f"Validating: {Path(file_path).name}")
        print(f"{'='*80}\n")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            return {"status": "ERROR", "message": f"Failed to load JSON: {str(e)}"}
        
        results = {
            "file": Path(file_path).name,
            "schema_validation": self._validate_schema(data),
            "functional_validation": self._validate_functions(data),
            "behavioral_validation": self._validate_behaviors(data),
            "structural_validation": self._validate_structures(data),
            "layout_validation": self._validate_layout(data),
            "score_validation": self._validate_scores(data),
            "cross_validation": self._cross_validate(data),
            "warnings": self.warnings.copy(),
            "errors": self.errors.copy()
        }
        
        # Calculate overall validation score
        results["overall_status"] = self._calculate_overall_status(results)
        
        # Clear for next file
        self.warnings.clear()
        self.errors.clear()
        
        return results
    
    def _validate_schema(self, data: Dict) -> Dict[str, Any]:
        """Validate required fields and data types"""
        print("[*] Schema Validation...")
        
        required_top_level = []
        optional_top_level = ["node_id", "prototype_id", "node_type", "generation_level", 
                             "parent_node_id", "transformation_type", "strategy_name", 
                             "reasoning", "prototype_name", "composite_score", 
                             "generation_method", "refinement_history"]
        
        # Check for common keys
        found_keys = set(data.keys())
        common_keys = found_keys & set(optional_top_level)
        
        missing = []
        for key in required_top_level:
            if key not in data:
                missing.append(key)
        
        # Check for essential sections
        essential_sections = ["functions", "behaviors", "structures", "layout", "scores"]
        missing_sections = [s for s in essential_sections if s not in data]
        
        if missing_sections:
            self.errors.append(f"Missing essential sections: {missing_sections}")
        
        return {
            "status": "PASS" if not missing_sections else "FAIL",
            "found_keys": list(common_keys),
            "missing_required": missing,
            "missing_sections": missing_sections
        }
    
    def _validate_functions(self, data: Dict) -> Dict[str, Any]:
        """Validate functional requirements"""
        print("[+] Functional Validation...")
        
        if "functions" not in data:
            return {"status": "SKIP", "reason": "No functions data"}
        
        functions = data["functions"]
        total_functions = len(functions) if isinstance(functions, dict) else len(functions)
        
        issues = []
        total_area_required = 0
        priority_distribution = []
        
        # Handle both dict and list formats
        func_items = functions.items() if isinstance(functions, dict) else enumerate(functions)
        
        for fid, func in func_items:
            # Check priority
            priority = func.get("priority", 0)
            priority_distribution.append(priority)
            
            if priority < 0 or priority > 1:
                issues.append(f"Invalid priority {priority} for {func.get('name', fid)}")
            
            # Check spatial requirements
            if "spatial_requirements" in func:
                sr = func["spatial_requirements"]
                min_area = sr.get("min_area", 0)
                pref_area = sr.get("preferred_area", 0)
                max_area = sr.get("max_area", 0)
                
                if not (min_area <= pref_area <= max_area):
                    issues.append(f"Invalid area range for {func.get('name', fid)}")
                
                total_area_required += pref_area
            elif "area_required" in func:
                total_area_required += func["area_required"]
        
        return {
            "status": "PASS" if not issues else "WARNING",
            "total_functions": total_functions,
            "total_area_required": round(total_area_required, 2),
            "avg_priority": round(sum(priority_distribution) / len(priority_distribution), 3) if priority_distribution else 0,
            "issues": issues
        }
    
    def _validate_behaviors(self, data: Dict) -> Dict[str, Any]:
        """Validate behavioral performance expectations"""
        print("[~] Behavioral Validation...")
        
        if "behaviors" not in data:
            return {"status": "SKIP", "reason": "No behaviors data"}
        
        behaviors = data["behaviors"]
        issues = []
        satisfaction_scores = []
        
        # Check each behavior category
        for bid, behavior in behaviors.items():
            # For expected behaviors
            if isinstance(behavior, dict) and "target_value" in behavior:
                target = behavior.get("target_value")
                actual = behavior.get("actual_value")
                tolerance = behavior.get("tolerance", 0)
                
                if actual is not None and target is not None:
                    # Calculate satisfaction
                    deviation = abs(actual - target) / target if target != 0 else 0
                    
                    if deviation > tolerance:
                        issues.append(f"{bid}: Deviation {deviation:.2%} exceeds tolerance {tolerance:.2%}")
                
                # Check for satisfaction scores in nested format
                if "actual" in behavior and isinstance(behavior["actual"], dict):
                    sat = behavior["actual"].get("satisfaction")
                    if sat is not None:
                        satisfaction_scores.append(sat)
        
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else None
        
        return {
            "status": "PASS" if not issues else "WARNING",
            "total_behaviors": len(behaviors),
            "avg_satisfaction": round(avg_satisfaction, 3) if avg_satisfaction else "N/A",
            "issues": issues
        }
    
    def _validate_structures(self, data: Dict) -> Dict[str, Any]:
        """Validate structural elements"""
        print("[#] Structural Validation...")
        
        if "structures" not in data:
            return {"status": "SKIP", "reason": "No structures data"}
        
        structures = data["structures"]
        issues = []
        
        # Typical U-value ranges (W/m²K) for validation
        u_value_ranges = {
            "external_wall": (0.15, 0.35),
            "internal_wall": (0.30, 0.60),
            "partition": (0.30, 0.60),
            "window": (0.70, 2.5),
            "floor": (0.15, 0.35),
            "roof": (0.10, 0.25),
            "slab": (0.15, 0.40)
        }
        
        for sid, struct in structures.items():
            struct_type = struct.get("structure_type", struct.get("type", "")).lower()
            
            # Check U-values if present
            if "thermal_properties" in struct:
                u_val = struct["thermal_properties"].get("U_value")
                if u_val and struct_type in u_value_ranges:
                    min_u, max_u = u_value_ranges[struct_type]
                    if not (min_u <= u_val <= max_u):
                        self.warnings.append(f"{sid}: U-value {u_val} outside typical range [{min_u}, {max_u}]")
            elif "u_value" in struct:
                u_val = struct["u_value"]
                if u_val and struct_type in u_value_ranges:
                    min_u, max_u = u_value_ranges.get(struct_type, (0, 3))
                    if not (min_u <= u_val <= max_u):
                        self.warnings.append(f"{sid}: U-value {u_val} outside typical range [{min_u}, {max_u}]")
        
        return {
            "status": "PASS" if not issues else "WARNING",
            "total_structures": len(structures),
            "issues": issues
        }
    
    def _validate_layout(self, data: Dict) -> Dict[str, Any]:
        """Validate layout metrics and spatial configuration"""
        print("[@] Layout Validation...")
        
        if "layout" not in data:
            return {"status": "SKIP", "reason": "No layout data"}
        
        layout = data["layout"]
        issues = []
        
        total_area = layout.get("total_area", 0)
        rooms = layout.get("rooms", {})
        
        # Calculate actual room areas
        room_area_sum = 0
        room_count = 0
        
        if isinstance(rooms, dict):
            for rid, room in rooms.items():
                area = room.get("area", 0)
                room_area_sum += area
                room_count += 1
                
                # Check if dimensions match area (if provided)
                if "width" in room and "height" in room:
                    calculated_area = room["width"] * room["height"]
                    if abs(calculated_area - area) > 0.1:
                        issues.append(f"{rid}: Area mismatch - declared: {area}, calculated: {calculated_area}")
        
        # Compare total area with sum of rooms
        if room_area_sum > 0:
            area_diff = abs(total_area - room_area_sum)
            area_diff_pct = (area_diff / total_area * 100) if total_area > 0 else 0
            
            if area_diff_pct > 25:  # Allow up to 25% difference for circulation
                self.warnings.append(f"Large area discrepancy: {area_diff_pct:.1f}% (likely circulation space)")
        
        # Validate layout metrics
        metrics = {}
        if "compactness" in layout:
            metrics["compactness"] = layout["compactness"]
            if not (0 <= layout["compactness"] <= 1):
                issues.append("Compactness score out of range [0, 1]")
        
        if "circulation_efficiency" in layout:
            metrics["circulation_efficiency"] = layout["circulation_efficiency"]
            if not (0 <= layout["circulation_efficiency"] <= 1):
                issues.append("Circulation efficiency out of range [0, 1]")
        
        if "adjacency_satisfaction" in layout:
            metrics["adjacency_satisfaction"] = layout["adjacency_satisfaction"]
            if not (0 <= layout["adjacency_satisfaction"] <= 1):
                issues.append("Adjacency satisfaction out of range [0, 1]")
        
        return {
            "status": "PASS" if not issues else "WARNING",
            "total_area": total_area,
            "room_area_sum": round(room_area_sum, 2),
            "room_count": room_count,
            "area_utilization": round((room_area_sum / total_area * 100), 1) if total_area > 0 else 0,
            "metrics": metrics,
            "issues": issues
        }
    
    def _validate_scores(self, data: Dict) -> Dict[str, Any]:
        """Validate scoring consistency"""
        print("[%] Score Validation...")
        
        if "scores" not in data:
            return {"status": "SKIP", "reason": "No scores data"}
        
        scores = data["scores"]
        issues = []
        
        # Extract individual scores
        individual_scores = []
        score_names = []
        
        for key, value in scores.items():
            if isinstance(value, (int, float)) and key != "composite_score":
                individual_scores.append(value)
                score_names.append(key)
                
                # Check range
                if not (0 <= value <= 1):
                    issues.append(f"{key} out of valid range: {value}")
        
        # Validate composite score calculation
        if "composite_score" in scores and individual_scores:
            expected_composite = sum(individual_scores) / len(individual_scores)
            actual_composite = scores["composite_score"]
            
            diff = abs(expected_composite - actual_composite)
            if diff > 0.01:  # Allow small floating point differences
                self.warnings.append(
                    f"Composite score mismatch - Expected: {expected_composite:.3f}, "
                    f"Actual: {actual_composite:.3f}, Diff: {diff:.3f}"
                )
        
        # Also check top-level composite_score if present
        if "composite_score" in data and "composite_score" in scores:
            if data["composite_score"] != scores["composite_score"]:
                issues.append("Top-level composite_score differs from scores.composite_score")
        
        return {
            "status": "PASS" if not issues else "WARNING",
            "individual_scores": {name: score for name, score in zip(score_names, individual_scores)},
            "composite_score": scores.get("composite_score"),
            "score_range": [round(min(individual_scores), 3), round(max(individual_scores), 3)] if individual_scores else None,
            "avg_score": round(sum(individual_scores) / len(individual_scores), 3) if individual_scores else None,
            "issues": issues
        }
    
    def _cross_validate(self, data: Dict) -> Dict[str, Any]:
        """Cross-validate between different sections"""
        print("[&] Cross Validation...")
        
        issues = []
        
        # Check if function areas match layout areas
        if "functions" in data and "layout" in data:
            func_data = data["functions"]
            layout_data = data["layout"]
            
            # Create mapping of function names to required areas
            func_areas = {}
            if isinstance(func_data, dict):
                for fid, func in func_data.items():
                    name = func.get("name", "").lower().replace("provide_", "")
                    if "spatial_requirements" in func:
                        func_areas[name] = func["spatial_requirements"].get("preferred_area", 0)
                    elif "area_required" in func:
                        func_areas[func.get("name", "")] = func["area_required"]
            
            # Check against layout rooms
            if "rooms" in layout_data and isinstance(layout_data["rooms"], dict):
                for rid, room in layout_data["rooms"].items():
                    room_name = room.get("name", "").lower()
                    room_area = room.get("area", 0)
                    
                    # Try to find matching function
                    matched = False
                    for func_name, req_area in func_areas.items():
                        if func_name in room_name or room_name.replace(" ", "_") == func_name:
                            matched = True
                            # Allow 10% tolerance
                            if abs(room_area - req_area) / req_area > 0.10:
                                self.warnings.append(
                                    f"Area mismatch for {room_name}: "
                                    f"Required={req_area}, Actual={room_area}"
                                )
                            break
        
        return {
            "status": "PASS" if not issues else "WARNING",
            "issues": issues
        }
    
    def _calculate_overall_status(self, results: Dict) -> str:
        """Calculate overall validation status"""
        
        if self.errors:
            return "[X] FAIL"
        elif self.warnings:
            return "[!] PASS WITH WARNINGS"
        else:
            return "[OK] PASS"
    
    def print_summary(self, results: Dict):
        """Print validation summary"""
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY: {results['file']}")
        print(f"{'='*80}\n")
        
        print(f"Overall Status: {results['overall_status']}\n")
        
        for section, data in results.items():
            if section in ["file", "overall_status", "warnings", "errors"]:
                continue
            
            if isinstance(data, dict) and "status" in data:
                status_icon = {"PASS": "[OK]", "WARNING": "[!]", "FAIL": "[X]", "SKIP": "[>]"}.get(data["status"], "[?]")
                print(f"{status_icon} {section.replace('_', ' ').title()}: {data['status']}")
                
                # Print key metrics
                for key, value in data.items():
                    if key not in ["status", "issues", "reason"]:
                        print(f"   • {key}: {value}")
        
        if results["warnings"]:
            print(f"\n[!] WARNINGS ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"   - {warning}")
        
        if results["errors"]:
            print(f"\n[X] ERRORS ({len(results['errors'])}):")
            for error in results["errors"]:
                print(f"   - {error}")
        
        print()


def validate_directory(directory_path: str):
    """Validate all JSON files in a directory"""
    validator = LayoutValidator()
    results_summary = []
    
    path = Path(directory_path)
    json_files = list(path.glob("*.json")) + list(path.glob("**/*.json"))
    
    print(f"\n{'#'*80}")
    print(f"FBSL-KAGS LAYOUT VALIDATION FRAMEWORK")
    print(f"Directory: {directory_path}")
    print(f"Found {len(json_files)} JSON files")
    print(f"{'#'*80}\n")
    
    for json_file in json_files:
        try:
            results = validator.validate_file(str(json_file))
            validator.print_summary(results)
            results_summary.append(results)
        except Exception as e:
            print(f"[X] Error validating {json_file.name}: {str(e)}\n")
    
    # Print overall summary
    print(f"\n{'#'*80}")
    print("OVERALL VALIDATION REPORT")
    print(f"{'#'*80}\n")
    
    total = len(results_summary)
    passed = sum(1 for r in results_summary if "[OK]" in r["overall_status"])
    warnings = sum(1 for r in results_summary if "[!]" in r["overall_status"])
    failed = sum(1 for r in results_summary if "[X]" in r["overall_status"])
    
    print(f"Total Files: {total}")
    print(f"[OK] Passed: {passed}")
    print(f"[!] Passed with Warnings: {warnings}")
    print(f"[X] Failed: {failed}")
    print()
    
    return results_summary


if __name__ == "__main__":
    # Validate the case_study directory
    case_study_dir = Path(__file__).parent
    validate_directory(str(case_study_dir))
