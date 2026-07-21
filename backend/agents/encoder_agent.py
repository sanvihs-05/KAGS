"""
Enhanced Encoder Agent that uses Finnish floor plan embeddings
to create FBSL problem nodes from user requirements

✅ CRITICAL FIX: Now uses LLM to extract rooms, areas, adjacencies from natural language
🔧 FIXED: LLM connection using FBSL.py proven pattern
"""

import logging
from typing import Dict, Any, List, Optional
import re
import os
import yaml
import json
import requests  # Use requests like FBSL.py does
import shutil
import subprocess
from ..core.fbsl_models import (
    FBSLLayoutNode, Function, Behavior, Structure, 
    FunctionCategory, BehaviorCategory, StructureType, NodeType, Layout, Room
)
from ..core.finnish_fbsl_mapper import FinnishFBSLMapper
from ..database.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class EncoderAgent:
    """
    Encoder Agent: Transforms user requirements into structured FBSL problem nodes
    
    ✅ NOW USES LLM TO EXTRACT: rooms, areas, adjacencies, behaviors from natural language
    🔧 FIXED: Uses requests library like FBSL.py for reliable Ollama connection
    """
    
    def __init__(self, 
                 vector_store: VectorStoreManager,
                 llm_model: Optional[str] = None,
                 llm_base_url: str = "http://localhost:11434"):
        
        self.vector_store = vector_store

        # Load LLM config from YAML if available
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config = {}
        try:
            with open(os.path.abspath(config_path), 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            config = {}

        llm_cfg = config.get('llm', {})
        self.llm_model = llm_model or llm_cfg.get('model') or 'llama3.1:8b'
        # Allow overriding Ollama URL via environment or config
        self.ollama_url = os.getenv('OLLAMA_BASE_URL') or llm_cfg.get('base_url') or llm_base_url

        # ── Provider selection ───────────────────────────────────────────
        # Three modes via KAGS_LLM_PROVIDER:
        #   'ollama'  — local only, no cloud attempt (old default behavior)
        #   'openai'  — cloud only, any OpenAI-compatible chat-completions API
        #               (OpenAI, Groq, OpenRouter, Together, Gemini's openai
        #               endpoint...), no local fallback
        #   'auto' / unset (DEFAULT) — try cloud first if a key is configured,
        #               fall back to Ollama on any cloud failure (timeout,
        #               connection error, bad response), fall back to the
        #               rule-based parser only if BOTH fail. This matters on
        #               VRAM-constrained hardware: measured on this machine,
        #               a cold Ollama model swap alone took 38s before
        #               generating a single token, blowing the old 60s
        #               timeout — a fast cloud provider avoids that entirely
        #               while Ollama stays available offline.
        #
        # Cloud config: KAGS_LLM_API_KEY, KAGS_LLM_BASE_URL, KAGS_LLM_MODEL.
        # Convenience: GROQ_API_KEY alone (no KAGS_LLM_* needed) auto-targets
        # Groq's free, fast (LPU) OpenAI-compatible endpoint.
        self.llm_provider = (
            os.getenv('KAGS_LLM_PROVIDER') or llm_cfg.get('provider') or 'auto'
        ).strip().lower()

        groq_key = os.getenv('GROQ_API_KEY')
        generic_key = os.getenv('KAGS_LLM_API_KEY') or llm_cfg.get('openai_api_key')
        if generic_key:
            self.openai_api_key = generic_key
            self.openai_base_url = (
                os.getenv('KAGS_LLM_BASE_URL') or llm_cfg.get('openai_base_url')
                or 'https://api.openai.com/v1'
            ).rstrip('/')
            default_cloud_model = llm_cfg.get('openai_model') or 'gpt-4o-mini'
        elif groq_key:
            self.openai_api_key = groq_key
            self.openai_base_url = (
                os.getenv('KAGS_LLM_BASE_URL') or 'https://api.groq.com/openai/v1'
            ).rstrip('/')
            default_cloud_model = 'llama-3.3-70b-versatile'
        else:
            self.openai_api_key = None
            self.openai_base_url = (
                os.getenv('KAGS_LLM_BASE_URL') or llm_cfg.get('openai_base_url')
                or 'https://api.openai.com/v1'
            ).rstrip('/')
            default_cloud_model = llm_cfg.get('openai_model') or 'gpt-4o-mini'

        self.cloud_model = os.getenv('KAGS_LLM_MODEL') or default_cloud_model
        try:
            self.llm_timeout = int(os.getenv('KAGS_LLM_TIMEOUT') or llm_cfg.get('timeout') or 60)
        except (TypeError, ValueError):
            self.llm_timeout = 60
        try:
            # Short: a cloud attempt should fail fast so auto mode still has
            # budget left to try Ollama within the same request.
            self.cloud_timeout = int(os.getenv('KAGS_LLM_CLOUD_TIMEOUT') or 20)
        except (TypeError, ValueError):
            self.cloud_timeout = 20

        self.cloud_available = bool(self.openai_api_key)
        if self.cloud_available:
            logger.info(
                f"✅ Cloud LLM configured: {self.openai_base_url} model={self.cloud_model}"
            )
        elif self.llm_provider == 'openai':
            logger.warning(
                "⚠️ KAGS_LLM_PROVIDER=openai but no API key set "
                "(KAGS_LLM_API_KEY or GROQ_API_KEY) - falling back to rule-based parser"
            )

        if self.llm_provider == 'openai':
            self.llm_available = self.cloud_available
        elif self.llm_provider == 'ollama':
            self.llm_available = self._test_llm_connection()
            if self.llm_available:
                logger.info(f"✅ Ollama LLM available. Using model {self.llm_model}")
            else:
                logger.warning("⚠️ Ollama LLM not available - falling back to rule-based parser")
        else:  # auto
            ollama_ok = self._test_llm_connection()
            self.llm_available = self.cloud_available or ollama_ok
            if not self.cloud_available and ollama_ok:
                logger.info(f"✅ Ollama LLM available (no cloud key set). Using model {self.llm_model}")
            elif not self.llm_available:
                logger.warning("⚠️ No LLM available (cloud or Ollama) - falling back to rule-based parser")
        
        if vector_store.finnish_embeddings:
            self.finnish_mapper = FinnishFBSLMapper(vector_store.finnish_embeddings)
        else:
            self.finnish_mapper = None
            logger.warning("Finnish embeddings not available, using basic encoding")
        
        logger.info(f"✓ Encoder Agent initialized with model={self.llm_model}")
    
    def _test_llm_connection(self) -> bool:
        """Test if Ollama is available and responding - using FBSL.py pattern"""
        # First, try HTTP endpoint if Ollama HTTP API is available
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m.get('name') for m in response.json().get('models', [])]
                logger.info(f"Available Ollama models (http): {models}")
                # prefer gemma if present
                for pref in ['gemma3:latest', 'gemma3:4b', 'gemma3:latest', 'gemma2:latest', 'llama3.1:8b']:
                    if pref in models:
                        self.llm_model = pref
                        logger.info(f"Auto-selected local model: {self.llm_model}")
                        break
                return True
        except Exception:
            logger.debug("Ollama HTTP API not reachable, trying CLI detection...")

        # Fallback: try `ollama` CLI to list available models and prefer gemma
        try:
            cli_exe = os.getenv('OLLAMA_CLI_PATH', 'ollama')
            try:
                proc = subprocess.run([cli_exe, 'list'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5)
                out = proc.stdout.strip()
            except Exception as e:
                logger.debug(f"Ollama CLI list invocation failed ({cli_exe}): {e}")
                out = ''

            models = []
            for line in out.splitlines():
                # skip header lines that contain NAME or ID
                if line.lower().startswith('name') or line.strip() == '':
                    continue
                parts = line.split()
                if parts:
                    models.append(parts[0].strip())
            logger.info(f"Available Ollama models (cli): {models}")
            # prefer gemma models
            for pref in ['gemma3:latest', 'gemma3:4b', 'gemma2:latest', 'llama3.1:8b']:
                if pref in models:
                    self.llm_model = pref
                    logger.info(f"Auto-selected local model: {self.llm_model}")
                    return True
            # if any models exist, keep configured or pick first
            if models:
                logger.info(f"Models present but preferred not found, keeping configured model {self.llm_model}")
                return True
        except Exception as e:
            logger.debug(f"Ollama CLI detection failed: {e}")

        return False
    
    def encode_requirements(self, user_input: str, context: Dict[str, Any] = None) -> FBSLLayoutNode:
        """
        Main encoding method: Transform user requirements into FBSL problem node
        
        ✅ NOW USES LLM TO EXTRACT SPATIAL PROGRAM
        
        GUARANTEES:
        - node.layout is never None
        - node.layout.rooms is never empty
        - Every function has at least one corresponding behavior
        - Every behavior has both target_value and actual_value set
        """
        logger.info(f"Encoding requirements: {user_input[:100]}...")
        
        # ✅ CRITICAL FIX: Use LLM to extract structured spatial program
        logger.info("🤖 Step 1: Using LLM to extract spatial program from requirements...")
        spatial_program = self._extract_spatial_program_with_llm(user_input, context)
        logger.info(f"✓ LLM extracted: {len(spatial_program.get('rooms', []))} rooms, "
                   f"{len(spatial_program.get('adjacencies', []))} adjacencies")
        
        # Step 2: Create base FBSL node from extracted program
        logger.info("🗂️ Step 2: Creating FBSL node from spatial program...")
        node = self._create_node_from_spatial_program(spatial_program, user_input)
        
        # ✅ CRITICAL DEBUG: Check node immediately after creation
        if node.layout and node.layout.rooms:
            logger.info(f"✓ Node created with {len(node.layout.rooms)} rooms")
        else:
            logger.error(f"❌ Node has NO ROOMS after _create_node_from_spatial_program!")
            logger.error(f"   node.layout = {node.layout}")
            if node.layout:
                logger.error(f"   node.layout.rooms = {getattr(node.layout, 'rooms', 'NO ROOMS ATTR')}")
        
        # Step 3: Add constraints from context
        if context:
            self._add_constraints(node, context)
        
        # Step 4: Validate FBSL consistency
        is_valid, issues = node.validate_fbsl_consistency()
        if not is_valid:
            logger.warning(f"FBSL validation issues: {issues}")
        
        # ✅ FINAL CHECK before returning
        if not node.layout or not node.layout.rooms or len(node.layout.rooms) == 0:
            logger.error("❌ FINAL CHECK FAILED: Node has no rooms before return!")
            logger.error(f"   Creating emergency fallback...")
            # Create absolute emergency fallback
            from ..core.fbsl_models import Layout, Room
            if not node.layout:
                node.layout = Layout()
            if not node.layout.rooms:
                node.layout.rooms = {}
            
            if len(node.layout.rooms) == 0:
                room = Room(
                    name="Emergency Living",
                    room_type="living",
                    room_number="1",
                    area=20.0,
                    height=3.0
                )
                room.calculate_volume()
                node.layout.rooms[room.room_id] = room
                node.layout.total_area = 20.0
                node.layout.used_area = 20.0
                node.layout.calculate_metrics()
                logger.info(f"✓ Emergency fallback created: 1 room")
        
        logger.info(
            f"✓ Encoded to FBSL node: {len(node.functions)} functions, "
            f"{len(node.behaviors)} behaviors, {len(node.structures)} structures, "
            f"{len(node.layout.rooms)} rooms"
        )
        
        return node
    
    def _extract_spatial_program_with_llm(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        ✅ NEW METHOD: Use LLM to extract structured spatial program
        🔧 FIXED: Uses requests library pattern from FBSL.py
        
        Returns:
            {
                'rooms': [{'type': str, 'name': str, 'area_min': float, 'area_max': float, 
                          'requirements': [str], 'orientation': str}, ...],
                'adjacencies': [{'room1': str, 'room2': str, 'type': str, 'strength': str}, ...],
                'constraints': [str],
                'priorities': [str]
            }
        """
        
        system_prompt = """You are an expert architectural programmer. Extract structured spatial information from user requirements.

Your task: Parse the text and return a JSON object with this EXACT structure:

{
  "rooms": [
    {
      "type": "bedroom|living_room|kitchen|bathroom|study|dining|utility|storage|hallway|balcony",
      "name": "descriptive name (e.g., 'Master Bedroom', 'Child Bedroom')",
      "area_min": minimum area in sqm (number),
      "area_max": maximum area in sqm (number),
      "requirements": ["list", "of", "requirements"],
      "orientation": "north|south|east|west|northeast|northwest|southeast|southwest|any"
    }
  ],
  "adjacencies": [
    {
      "room1": "room type",
      "room2": "room type", 
      "type": "required|preferred|avoid",
      "strength": "high|medium|low"
    }
  ],
  "constraints": ["list of design constraints"],
  "priorities": ["list of priorities"]
}

CRITICAL RULES:
1. Extract ALL rooms mentioned
2. Extract area ranges (if "13-15 sqm" then min=13, max=15)
3. Identify adjacencies from phrases like "connected to", "next to", "grouped with"
4. Return ONLY valid JSON, no markdown, no explanations
5. Use standard room types from the list above"""

        user_prompt = f"""Requirements:
{user_input}

Context: {json.dumps(context) if context else 'None'}

Extract all spatial information and return as JSON."""

        env_model = os.getenv('OLLAMA_MODEL') or os.getenv('LLM_MODEL')
        if env_model:
            self.llm_model = env_model
            logger.info(f"OLLAMA_MODEL override detected, using model {self.llm_model}")

        use_cli_force = os.getenv('OLLAMA_USE_CLI', '').lower() in ('1', 'true', 'yes')

        # ── Build the provider attempt chain ────────────────────────────
        # 'openai'/'ollama' pin to a single provider (no fallback between
        # them — an explicit choice should behave predictably for testing).
        # Default 'auto' tries cloud first (fast, no VRAM contention) and
        # falls back to Ollama on ANY cloud failure, so a flaky/rate-limited
        # API never blocks a design that local Ollama could still produce.
        if self.llm_provider == 'openai':
            attempts = ['cloud']
        elif self.llm_provider == 'ollama':
            attempts = ['ollama']
        else:
            attempts = (['cloud'] if self.cloud_available else []) + ['ollama']

        for kind in attempts:
            try:
                if kind == 'cloud':
                    if not self.cloud_available:
                        raise RuntimeError('no cloud API key configured')
                    response_text = self._call_cloud_llm(system_prompt, user_prompt)
                else:
                    if not self.llm_available and not use_cli_force:
                        raise RuntimeError('Ollama unavailable')
                    response_text = self._call_ollama_llm(
                        system_prompt, user_prompt, use_cli_force
                    )

                spatial_program = self._extract_json_from_llm_response(response_text)
                if not spatial_program or not spatial_program.get('rooms'):
                    raise RuntimeError(
                        f"empty/invalid JSON from {kind} LLM: {response_text[:300]!r}"
                    )

                spatial_program = self._validate_spatial_program(spatial_program)
                logger.info(f"  ✓ {kind} LLM extracted {len(spatial_program['rooms'])} rooms")
                logger.debug(f"     Rooms: {[r['name'] for r in spatial_program['rooms']]}")
                return spatial_program

            except Exception as e:
                remaining = attempts[attempts.index(kind) + 1:]
                logger.warning(
                    f"  ✗ {kind} extraction failed: {e}"
                    + (f" — trying {remaining[0]}" if remaining else "")
                )
                continue

        logger.warning("  → All LLM providers failed - using rule-based fallback parser")
        return self._fallback_parse(user_input)

    def _call_cloud_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call any OpenAI-compatible chat-completions endpoint. Raises on
        any failure (timeout, connection error, non-200, malformed body) so
        the caller's fallback chain can move to the next provider."""
        headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self.cloud_model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'temperature': 0.1,
        }
        response = requests.post(
            f"{self.openai_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.cloud_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"cloud LLM API returned status {response.status_code}: {response.text[:400]}"
            )
        return response.json()['choices'][0]['message']['content']

    def _call_ollama_llm(self, system_prompt: str, user_prompt: str, use_cli_force: bool) -> str:
        """Call local Ollama via HTTP (or CLI if forced). Raises on any
        failure so the caller's fallback chain can proceed to the rule-based
        parser."""
        if use_cli_force:
            cli_exe = os.getenv('OLLAMA_CLI_PATH', 'ollama')
            try:
                cmd = [cli_exe, 'run', self.llm_model, f"{system_prompt}\n\n{user_prompt}"]
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                                      errors='replace', timeout=self.llm_timeout)
                if proc.returncode != 0:
                    raise RuntimeError(f"Ollama CLI failed: {proc.stderr[:400]}")
                return proc.stdout
            except FileNotFoundError:
                raise RuntimeError(f"Ollama CLI binary not found: {cli_exe}")

        payload = {
            "model": self.llm_model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": 0.1},
        }
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=self.llm_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API returned status {response.status_code}: {response.text[:400]}")
        return response.json().get("response", "")
    
    def _extract_json_from_llm_response(self, text: str) -> Dict:
        """Extract JSON from LLM response, handling markdown and other formatting"""
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode failed: {e}")
        
        return {}
    
    def _validate_spatial_program(self, program: Dict) -> Dict:
        """Validate and normalize spatial program from LLM"""
        
        # Ensure required keys exist
        validated = {
            'rooms': program.get('rooms', []),
            'adjacencies': program.get('adjacencies', []),
            'constraints': program.get('constraints', []),
            'priorities': program.get('priorities', [])
        }
        
        # Validate rooms
        valid_rooms = []
        for room in validated['rooms']:
            if isinstance(room, dict) and 'type' in room:
                # Normalize room type - FIX: Remove pipes and special characters
                room_type = str(room['type']).lower().replace(' ', '_')
                # Remove pipes, slashes, parentheses
                room_type = room_type.split('|')[0].split('/')[0].split('(')[0].strip()
                # Remove trailing underscores
                room_type = room_type.rstrip('_')
                
                # Ensure area values
                area_min = float(room.get('area_min', 10.0))
                area_max = float(room.get('area_max', area_min * 1.5))
                
                # Clean name too
                name = room.get('name', room_type.replace('_', ' ').title())
                name = name.split('|')[0].split('/')[0].split('(')[0].strip()
                
                valid_rooms.append({
                    'type': room_type,
                    'name': name,
                    'area_min': area_min,
                    'area_max': area_max,
                    'area_preferred': (area_min + area_max) / 2,
                    'requirements': room.get('requirements', []),
                    'orientation': room.get('orientation', 'any')
                })
        
        validated['rooms'] = valid_rooms
        
        # Ensure at least one room
        if not validated['rooms']:
            logger.warning("LLM returned no rooms, adding default")
            validated['rooms'] = [{
                'type': 'living_room',
                'name': 'Living Room',
                'area_min': 15.0,
                'area_max': 25.0,
                'area_preferred': 20.0,
                'requirements': [],
                'orientation': 'any'
            }]
        
        return validated
    
    def _create_node_from_spatial_program(self, program: Dict, original_query: str) -> FBSLLayoutNode:
        """
        Create FBSL node from extracted spatial program
        
        ✅ This replaces the direct call to Finnish mapper
        """
        
        node = FBSLLayoutNode(node_type=NodeType.PROBLEM, generation_level=0)
        layout = Layout()
        layout.configuration_name = "LLM Extracted Layout"
        
        # Create Functions, Behaviors, Structures, and Rooms from program
        for i, room_data in enumerate(program['rooms']):
            room_type = room_data['type']
            room_name = room_data['name']
            area_preferred = room_data['area_preferred']
            area_min = room_data['area_min']
            area_max = room_data['area_max']
            requirements = room_data.get('requirements', [])
            orientation = room_data.get('orientation', 'any')
            
            # Create Function
            function = Function(
                name=f"provide_{room_type}",
                category=self._get_function_category(room_type),
                description=f"{room_name}: {', '.join(requirements[:3]) if requirements else 'general use'}",
                priority=self._get_room_priority(room_type),
                activities=self._get_room_activities(room_type),
                spatial_requirements={
                    'preferred_area': area_preferred,
                    'min_area': area_min,
                    'max_area': area_max,
                    'orientation': orientation
                }
            )
            node.add_function(function)
            
            # Create Behaviors (area + specific requirements)
            area_behavior = Behavior(
                category=BehaviorCategory.SPATIAL,
                metric_name=f"{room_type}_area",
                metric_unit="sqm",
                target_value=area_preferred,
                actual_value=area_preferred * 0.9,  # Initial estimate
                tolerance=0.2,
                derived_from_function=function.function_id
            )
            node.add_behavior(area_behavior)
            
            # Add requirement-specific behaviors
            for req in requirements:
                req_lower = req.lower()
                if 'ventilation' in req_lower or 'cross-ventilation' in req_lower:
                    node.add_behavior(Behavior(
                        category=BehaviorCategory.VENTILATION,
                        metric_name=f"{room_type}_ventilation",
                        metric_unit="ACH",
                        target_value=0.5,
                        actual_value=0.4,
                        tolerance=0.3,
                        derived_from_function=function.function_id
                    ))
                
                if 'light' in req_lower or 'daylight' in req_lower:
                    node.add_behavior(Behavior(
                        category=BehaviorCategory.LIGHTING,
                        metric_name=f"{room_type}_daylight",
                        metric_unit="%",
                        target_value=2.0,
                        actual_value=1.8,
                        tolerance=0.3,
                        derived_from_function=function.function_id
                    ))
            
            # Create Structure (partition + windows for habitable rooms)
            structure = Structure(
                name=f"{room_type}_partition",
                structure_type=StructureType.PARTITION,
                material_type="gypsum_board",
                category="partition",
                dimensions={'thickness': 0.1}
            )
            node.add_structure(structure)

            _habitable = {'bedroom', 'living_room', 'kitchen', 'dining', 'study', 'bathroom'}
            if room_type in _habitable:
                _window_ratios = {
                    'living_room': 0.25, 'bedroom': 0.18, 'kitchen': 0.20,
                    'study': 0.18, 'dining': 0.20, 'bathroom': 0.10,
                }
                window = Structure(
                    name=f"{room_type}_window",
                    structure_type=StructureType.WALL,
                    material_type="glazing",
                    category="envelope",
                    dimensions={
                        'window_ratio': _window_ratios.get(room_type, 0.15),
                        'thickness': 0.006
                    },
                    load_bearing=False
                )
                node.add_structure(window)
            
            # ✅ Create Room
            room = Room(
                name=room_name,
                room_type=room_type,
                room_number=str(i + 1),
                function_id=function.function_id,
                area=area_preferred,
                height=3.0
            )
            room.calculate_volume()
            layout.rooms[room.room_id] = room
            
            logger.debug(f"  → Created: {room_name} ({area_preferred:.1f} m²)")
        
        # Add node-level structures: HVAC (fixes ventilation S_b) + foundation (fixes S_s load-bearing penalty)
        hvac = Structure(
            name="hvac_ventilation_system",
            structure_type=StructureType.MEP,
            material_type="steel",
            category="services",
            dimensions={'duct_diameter': 0.3, 'flow_rate': 0.5},
            load_bearing=False
        )
        node.add_structure(hvac)

        foundation = Structure(
            name="reinforced_concrete_foundation",
            structure_type=StructureType.FOUNDATION,
            material_type="concrete",
            category="structural",
            dimensions={'thickness': 0.3, 'depth': 0.6},
            load_bearing=True
        )
        node.add_structure(foundation)

        # Calculate layout metrics
        layout.total_area = sum(r.area for r in layout.rooms.values())
        layout.used_area = layout.total_area
        layout.calculate_metrics()
        
        # Store adjacencies in metadata
        node.metadata['required_adjacencies'] = program.get('adjacencies', [])
        node.metadata['design_constraints'] = program.get('constraints', [])
        node.metadata['design_priorities'] = program.get('priorities', [])
        
        # Assign layout to node
        node.layout = layout
        
        logger.info(f"✓ Created node from spatial program: {len(layout.rooms)} rooms, {layout.total_area:.1f} m²")
        
        # ✅ CRITICAL DEBUG: Verify layout is attached
        if not node.layout or not node.layout.rooms:
            logger.error(f"❌ CRITICAL BUG: Layout lost after assignment!")
            logger.error(f"   node.layout = {node.layout}")
            logger.error(f"   layout object = {layout}")
            logger.error(f"   layout.rooms = {getattr(layout, 'rooms', 'NO ROOMS ATTR')}")
        else:
            logger.debug(f"✓ Verified: node.layout has {len(node.layout.rooms)} rooms")
        
        return node
    
    def _get_function_category(self, room_type: str) -> FunctionCategory:
        """Map room type to function category"""
        mapping = {
            'bedroom': FunctionCategory.SPATIAL,
            'living_room': FunctionCategory.SOCIAL,
            'kitchen': FunctionCategory.TECHNICAL,
            'bathroom': FunctionCategory.TECHNICAL,
            'study': FunctionCategory.SPATIAL,
            'dining': FunctionCategory.SOCIAL,
            'utility': FunctionCategory.TECHNICAL,
            'storage': FunctionCategory.SPATIAL,
            'hallway': FunctionCategory.SPATIAL,
            'balcony': FunctionCategory.ENVIRONMENTAL
        }
        return mapping.get(room_type, FunctionCategory.SPATIAL)
    
    def _get_room_priority(self, room_type: str) -> float:
        """Get priority for room type"""
        priorities = {
            'bedroom': 0.9,
            'kitchen': 0.95,
            'bathroom': 0.9,
            'living_room': 0.85,
            'study': 0.7,
            'dining': 0.75,
            'utility': 0.6,
            'storage': 0.5,
            'hallway': 0.65,
            'balcony': 0.6
        }
        return priorities.get(room_type, 0.7)
    
    def _get_room_activities(self, room_type: str) -> List[str]:
        """Get typical activities for room type"""
        activities = {
            'bedroom': ['sleeping', 'resting', 'privacy'],
            'living_room': ['socializing', 'relaxation', 'entertainment'],
            'kitchen': ['cooking', 'food_preparation', 'dining'],
            'bathroom': ['bathing', 'hygiene', 'sanitation'],
            'study': ['work', 'concentration', 'reading'],
            'dining': ['dining', 'socializing'],
            'utility': ['laundry', 'cleaning', 'storage'],
            'storage': ['storage', 'organization'],
            'hallway': ['circulation', 'transition'],
            'balcony': ['outdoor_access', 'fresh_air', 'relaxation']
        }
        return activities.get(room_type, ['general_use'])
    
    # Word → number for counts like "three bedrooms" (the old regex only read digits)
    _WORD_NUM = {
        'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'single': 1, 'double': 2, 'triple': 3,
    }

    # (synonym regex, canonical type, (default_min, default_max), countable?)
    _LEXICON = [
        (r'bedrooms?', 'bedroom', (12.0, 16.0), True),
        (r'bathrooms?', 'bathroom', (4.0, 8.0), True),
        (r'powder\s*rooms?|toilets?|\bwc\b|water\s*closets?', 'toilet', (2.0, 4.0), True),
        (r'kitchens?', 'kitchen', (14.0, 18.0), False),
        (r'living(?:\s*/?\s*dining)?(?:\s*(?:rooms?|areas?))?|great\s*rooms?|lounges?',
         'living_room', (30.0, 45.0), False),
        (r'dining(?:\s*(?:rooms?|areas?))?', 'dining', (10.0, 16.0), False),
        (r'home\s*offices?|offices?|stud(?:y|ies)|studios?', 'office', (10.0, 14.0), False),
        (r'laundr(?:y|ies)|utility(?:\s*rooms?)?', 'laundry', (4.0, 10.0), False),
        (r'mud\s*rooms?', 'mudroom', (4.0, 8.0), False),
        (r'garages?', 'garage', (18.0, 40.0), False),
        (r'store\s*rooms?|storages?|pantr(?:y|ies)|closets?', 'storage', (4.0, 12.0), False),
    ]

    @classmethod
    def _canonical_room_type(cls, mention: str) -> Optional[str]:
        """Map a matched room mention (e.g. 'dining area') to its canonical type."""
        mention = mention.strip().lower()
        for syn, rtype, _area, _countable in cls._LEXICON:
            if re.fullmatch(syn, mention):
                return rtype
        return None

    def _extract_adjacencies(self, text: str, present_types: set) -> List[Dict[str, str]]:
        """Extract adjacency requirements from connective phrases — the L in
        FBSL. Without this, briefs parsed by the fallback path carried ZERO
        adjacency requirements and the layout stage had nothing to satisfy.

        Handles: 'kitchen ... connected to the dining area', 'mudroom that
        connects to the garage', 'master bedroom with attached bathroom',
        'open-plan kitchen and living room', and avoid-phrases
        ('bedrooms separated from the living area').
        """
        syn_alt = '|'.join(f'(?:{syn})' for syn, _t, _a, _c in self._LEXICON)
        connect = r'(?:connect(?:ed|s)?\s+(?:directly\s+)?to|adjacent\s+to|next\s+to|attached\s+to|linked\s+to|beside|opening\s+(?:on|in)?to)'
        avoid = r'(?:separated?\s+from|away\s+from|far\s+from|isolated\s+from)'

        pairs = []

        def _add(a_txt, b_txt, kind):
            a, b = self._canonical_room_type(a_txt), self._canonical_room_type(b_txt)
            if not a or not b or a == b:
                return
            if a not in present_types or b not in present_types:
                return
            key = (min(a, b), max(a, b), kind)
            if key in {(min(p['room1'], p['room2']), max(p['room1'], p['room2']), p['type'])
                       for p in pairs}:
                return
            pairs.append({'room1': a, 'room2': b, 'type': kind, 'strength': 'high'})

        # "A ... connected to (the) B" — allow qualifier words in between
        # (e.g. "kitchen of 16 sqm connected to the dining area",
        #  "mudroom that connects to the garage")
        for m in re.finditer(
                r'\b(?P<a>' + syn_alt + r')\b[^.;,]{0,40}?' + connect +
                r'\s+(?:the\s+|a\s+|an\s+)?(?P<b>' + syn_alt + r')\b', text):
            _add(m.group('a'), m.group('b'), 'required')

        # "A with (an) attached B" / "A with ensuite B"
        for m in re.finditer(
                r'\b(?P<a>' + syn_alt + r')\b[^.;,]{0,40}?with\s+(?:an?\s+)?'
                r'(?:attached|ensuite|en-suite|adjoining)\s+(?P<b>' + syn_alt + r')\b', text):
            _add(m.group('a'), m.group('b'), 'required')

        # "open-plan A and B" / "combined A and B"
        for m in re.finditer(
                r'(?:open[-\s]plan|combined)\s+(?P<a>' + syn_alt + r')\s+and\s+'
                r'(?P<b>' + syn_alt + r')\b', text):
            _add(m.group('a'), m.group('b'), 'required')

        # avoid-phrases
        for m in re.finditer(
                r'\b(?P<a>' + syn_alt + r')\b[^.;,]{0,40}?' + avoid +
                r'\s+(?:the\s+|a\s+|an\s+)?(?P<b>' + syn_alt + r')\b', text):
            _add(m.group('a'), m.group('b'), 'avoid')

        return pairs

    @staticmethod
    def _area_after(text: str, pos: int) -> Optional[tuple]:
        """Find an explicit area spec just after a room mention, e.g. '(40 sqm)',
        '14 sqm each', '35-45 sqm'. Returns (area_min, area_max) or None."""
        window = text[pos:pos + 16]  # tight: only an area immediately adjacent to the room
        unit = r'(?:sqm|sq\s*m|m2|m²|square\s*met(?:er|re)s?)'
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*' + unit, window)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(r'(\d+(?:\.\d+)?)\s*' + unit, window)
        if m:
            a = float(m.group(1))
            return round(a * 0.85, 1), round(a * 1.15, 1)
        return None

    def _fallback_parse(self, user_input: str) -> Dict[str, Any]:
        """
        Deterministic fallback parser used when the LLM is unavailable or times
        out. Rewritten to fix the degenerate-program bug: the old version only
        matched DIGIT-prefixed rooms (missing "three bedrooms", "master
        bedroom", "ensuite bathroom") and ignored explicit areas.

        Now handles word-numbers, qualified singletons, and per-room areas.
        """
        parsed = {'rooms': [], 'adjacencies': [], 'constraints': [], 'priorities': []}
        text = ' ' + user_input.lower() + ' '
        num_alt = '|'.join(self._WORD_NUM.keys())
        qual = r'(?:master|guest|shared|ensuite|en-suite|main|primary|family|kids?|children\'?s?|junior|powder)'

        LEXICON = self._LEXICON

        def add_room(rtype, area):
            lo, hi = area
            parsed['rooms'].append({
                'type': rtype,
                'name': rtype.replace('_', ' ').title(),
                'area_min': lo, 'area_max': hi, 'area_preferred': round((lo + hi) / 2, 1),
                'requirements': [], 'orientation': 'any',
            })

        for syn, rtype, default_area, countable in LEXICON:
            if countable:
                # A "N-bedroom" / "N bedroom home" headline is authoritative — humans
                # read "4-bedroom home with master bedroom and three bedrooms" as 4,
                # not 8. Use the headline count if present; else sum enumerated mentions.
                head = re.search(r'\b(\d+|' + num_alt + r')[\s-]+(?:' + syn + r')', text)
                if head:
                    qtok = head.group(1)
                    total = int(qtok) if qtok.isdigit() else self._WORD_NUM.get(qtok, 1)
                    area = self._area_after(text, head.end()) or default_area
                else:
                    rx = re.compile(
                        r'(?:\b(?P<qty>\d+|' + num_alt + r')\s+)?'
                        r'(?:' + qual + r'\s+)?(?:' + syn + r')')
                    total, area = 0, None
                    for m in rx.finditer(text):
                        qtok = m.group('qty')
                        total += int(qtok) if (qtok and qtok.isdigit()) else self._WORD_NUM.get(qtok, 1)
                        if area is None:
                            area = self._area_after(text, m.end())
                    area = area or default_area
                for _ in range(min(total, 12)):   # cap runaway matches
                    add_room(rtype, area)
            else:
                m = re.search(r'(?:' + syn + r')', text)
                if m:
                    add_room(rtype, self._area_after(text, m.end()) or default_area)

        # Ensure at least one room
        if not parsed['rooms']:
            logger.warning("Fallback parser found no rooms - adding default living room")
            add_room('living_room', (15.0, 25.0))

        # Adjacency requirements (the L in FBSL) — extracted from connective
        # phrases so the layout stage has real requirements to satisfy.
        present = {r['type'] for r in parsed['rooms']}
        parsed['adjacencies'] = self._extract_adjacencies(text, present)

        logger.info(
            f"  ✓ Fallback parser extracted {len(parsed['rooms'])} rooms: "
            f"{[r['type'] for r in parsed['rooms']]}; "
            f"{len(parsed['adjacencies'])} adjacencies: "
            f"{[(a['room1'], a['room2'], a['type']) for a in parsed['adjacencies']]}"
        )
        return parsed
    
    def _add_constraints(self, node: FBSLLayoutNode, context: Dict[str, Any]):
        """Add constraints from context"""
        
        if 'site_area' in context:
            node.metadata['site_area'] = context['site_area']
        
        if 'site_boundary' in context:
            node.metadata['site_boundary'] = context['site_boundary']
        
        if 'budget' in context:
            node.metadata['budget_constraint'] = context['budget']
        
        if 'building_codes' in context:
            node.metadata['building_codes'] = context['building_codes']
    
    def encode_batch(self, requirements_list: List[str]) -> List[FBSLLayoutNode]:
        """Encode multiple requirements in batch"""
        nodes = []
        for req in requirements_list:
            try:
                node = self.encode_requirements(req)
                nodes.append(node)
            except Exception as e:
                logger.error(f"Failed to encode requirement: {e}")
        return nodes


def test_encoder_agent():
    """Test the encoder agent with LLM extraction"""
    from ..database.vector_store import VectorStoreManager
    
    print("🔧 Testing Encoder Agent with LLM Extraction")
    print("=" * 60)
    
    # Initialize
    vs = VectorStoreManager()
    encoder = EncoderAgent(vs)
    
    # Test cases
    test_cases = [
        "2 bedroom apartment with kitchen, bathroom, and living room",
        "Master bedroom 13-15 sqm with attached bathroom and cross-ventilation",
        "Living and dining area 20-22 sqm bright from north side",
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_input}")
        print("-" * 60)
        
        try:
            node = encoder.encode_requirements(test_input)
            
            print(f"✓ FBSL Node Created:")
            print(f"  • Node ID: {node.node_id[:8]}...")
            print(f"  • Functions: {len(node.functions)}")
            for func in list(node.functions.values())[:3]:
                print(f"    - {func.name} (priority: {func.priority:.2f})")
            
            print(f"  • Behaviors: {len(node.behaviors)}")
            for behav in list(node.behaviors.values())[:3]:
                print(f"    - {behav.metric_name}: {behav.target_value} {behav.metric_unit}")
            
            print(f"  • Rooms: {len(node.layout.rooms)}")
            for room in list(node.layout.rooms.values())[:3]:
                print(f"    - {room.name}: {room.area:.1f} m²")
            
            # Validate
            is_valid, issues = node.validate_fbsl_consistency()
            if is_valid:
                print(f"  ✓ FBSL validation passed")
            else:
                print(f"  ⚠ Validation issues: {issues}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n✅ Encoder Agent testing complete!")


if __name__ == "__main__":
    test_encoder_agent()