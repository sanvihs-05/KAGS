import psycopg2
from psycopg2.extras import RealDictCursor, Json
import yaml
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """PostgreSQL database manager for FBSL prototype storage"""
    
    def __init__(self, config_path: str = None, skip_schema_init: bool = False):
        from pathlib import Path

        if config_path is None:
            file_path = Path(__file__).resolve()
            candidates = [
                file_path.parents[1] / "config" / "config.yaml",
                file_path.parents[2] / "config" / "config.yaml",
            ]
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            if found is None:
                # As a last resort try cwd/config/config.yaml
                cwd_candidate = Path.cwd() / "config" / "config.yaml"
                if cwd_candidate.exists():
                    found = cwd_candidate
            if found is None:
                raise FileNotFoundError(f"Database config file not found. Tried: {', '.join(str(p) for p in candidates+[cwd_candidate])}")
            config_path = found
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Database config file not found at: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.db_config = config['database']
        self.conn = None
        self.connect()
        
        if not skip_schema_init:
            try:
                self.initialize_schema()
            except PermissionError:
                # If permission denied, log warning but don't crash
                # User can create tables manually or grant permissions
                logger.warning(
                    "⚠️  Schema initialization skipped due to permissions. "
                    "Database operations may fail. See database/setup_permissions.sql for help."
                )
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            logger.info("✓ Database connected successfully")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    def _sanitize_data(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        try:
            import numpy as _np
        except Exception:
            _np = None

        # Numpy types
        if _np is not None:
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
                return obj.item()
            if isinstance(obj, _np.generic):
                return obj.item()

        # Enums, datetimes, UUIDs
        try:
            from enum import Enum as _Enum
            import datetime as _dt
            import uuid as _uuid
        except Exception:
            _Enum = None
            _dt = None
            _uuid = None

        if _Enum is not None and isinstance(obj, _Enum):
            return obj.value
        if _dt is not None and isinstance(obj, _dt.datetime):
            return obj.isoformat()
        if _uuid is not None and isinstance(obj, _uuid.UUID):
            return str(obj)

        if isinstance(obj, dict):
            return {k: self._sanitize_data(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_data(v) for v in obj]
        return obj
    
    def initialize_schema(self):
        """Create database tables if they don't exist"""
        try:
            # First, check if we have CREATE privileges
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT has_schema_privilege(current_user, 'public', 'CREATE') as can_create;
                """)
                result = cur.fetchone()
                can_create = result[0] if result else False
                
                if not can_create:
                    logger.warning(
                        "⚠️  User does not have CREATE privileges on public schema. "
                        "Tables may already exist, or you need to grant privileges.\n"
                        "To fix, run as PostgreSQL superuser:\n"
                        "  GRANT CREATE ON SCHEMA public TO fbsl_user;\n"
                        "  GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fbsl_user;\n"
                        "Or create tables manually using the SQL in database/schema.sql"
                    )
                    # Try to check if tables already exist
                    cur.execute("""
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('fbsl_nodes', 'projects', 'evaluations');
                    """)
                    existing = [row[0] for row in cur.fetchall()]
                    if existing:
                        logger.info(f"✓ Found existing tables: {', '.join(existing)}")
                        return  # Tables exist, we can proceed
                    else:
                        raise PermissionError(
                            "Cannot create tables: insufficient privileges. "
                            "Please grant CREATE privileges or create tables manually."
                        )
            
            with self.conn.cursor() as cur:
                # FBSL Nodes table (prototypes, problems, evaluations)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fbsl_nodes (
                        node_id VARCHAR(255) PRIMARY KEY,
                        project_id VARCHAR(255),
                        parent_node_id VARCHAR(255),
                        children_ids JSONB,
                        node_type VARCHAR(50) NOT NULL,
                        generation_level INTEGER DEFAULT 0,
                        iteration_number INTEGER DEFAULT 1,
                        
                        -- FBSL Components (stored as JSONB)
                        functions JSONB,
                        behaviors JSONB,
                        structures JSONB,
                        layout JSONB,
                        
                        -- Scores
                        functional_score FLOAT DEFAULT 0.0,
                        behavioral_score FLOAT DEFAULT 0.0,
                        structural_score FLOAT DEFAULT 0.0,
                        layout_score FLOAT DEFAULT 0.0,
                        sustainability_score FLOAT DEFAULT 0.0,
                        composite_score FLOAT DEFAULT 0.0,
                        
                        -- Constraints and violations
                        constraints_satisfied JSONB,
                        violations JSONB,
                        
                        -- Metadata
                        metadata JSONB,
                        
                        -- Timestamps
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Indexes for common queries
                        CONSTRAINT fk_parent FOREIGN KEY (parent_node_id) 
                            REFERENCES fbsl_nodes(node_id) ON DELETE SET NULL
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_node_type ON fbsl_nodes(node_type);
                    CREATE INDEX IF NOT EXISTS idx_project_id ON fbsl_nodes(project_id);
                    CREATE INDEX IF NOT EXISTS idx_composite_score ON fbsl_nodes(composite_score DESC);
                    CREATE INDEX IF NOT EXISTS idx_created_at ON fbsl_nodes(created_at DESC);
                """)
                
                # Projects table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS projects (
                        project_id VARCHAR(255) PRIMARY KEY,
                        project_name VARCHAR(255) NOT NULL,
                        description TEXT,
                        requirements TEXT,
                        context JSONB,
                        status VARCHAR(50) DEFAULT 'active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_project_status ON projects(status);
                """)
                
                # Evaluations table (detailed evaluation results)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS evaluations (
                        evaluation_id VARCHAR(255) PRIMARY KEY,
                        node_id VARCHAR(255) NOT NULL,
                        project_id VARCHAR(255),
                        
                        -- Detailed scores
                        functional_adequacy JSONB,
                        behavioral_performance JSONB,
                        structural_feasibility JSONB,
                        layout_efficiency JSONB,
                        sustainability JSONB,
                        
                        -- Overall assessment
                        composite_score FLOAT,
                        rank INTEGER,
                        
                        -- Qualitative assessment
                        strengths JSONB,
                        weaknesses JSONB,
                        recommendations JSONB,
                        
                        -- Timestamp
                        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        CONSTRAINT fk_node FOREIGN KEY (node_id) 
                            REFERENCES fbsl_nodes(node_id) ON DELETE CASCADE,
                        CONSTRAINT fk_project FOREIGN KEY (project_id) 
                            REFERENCES projects(project_id) ON DELETE SET NULL
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_eval_node ON evaluations(node_id);
                    CREATE INDEX IF NOT EXISTS idx_eval_project ON evaluations(project_id);
                    CREATE INDEX IF NOT EXISTS idx_eval_score ON evaluations(composite_score DESC);
                """)
                
                self.conn.commit()
                logger.info("✓ Database schema initialized")
        except PermissionError:
            # Re-raise permission errors
            raise
        except Exception as e:
            # Check if it's a "relation already exists" error (that's okay)
            error_str = str(e).lower()
            if 'already exists' in error_str or 'duplicate' in error_str:
                logger.info("✓ Tables already exist, schema is ready")
                self.conn.rollback()
            else:
                logger.error(f"✗ Schema initialization failed: {e}")
                self.conn.rollback()
                # Don't raise - allow connection to proceed if tables might exist
                logger.warning("⚠️  Continuing anyway - tables may already exist")
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute a query"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            self.conn.commit()
            return cur.fetchall() if cur.description else None
    
    def store_fbsl_node(self, node_dict: Dict[str, Any]) -> bool:
        """
        Store an FBSL node in the database
        
        Args:
            node_dict: Dictionary representation of FBSLLayoutNode (from to_dict())
        
        Returns:
            True if successful
        """
        try:
            # Ensure parent_node_id exists to satisfy FK constraint; if not, null it out
            parent_id = node_dict.get('parent_node_id')
            if parent_id:
                with self.conn.cursor() as _cur:
                    _cur.execute("SELECT 1 FROM fbsl_nodes WHERE node_id = %s", (parent_id,))
                    exists = _cur.fetchone() is not None
                if not exists:
                    node_dict['parent_node_id'] = None

            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO fbsl_nodes (
                        node_id, project_id, parent_node_id, children_ids,
                        node_type, generation_level, iteration_number,
                        functions, behaviors, structures, layout,
                        functional_score, behavioral_score, structural_score,
                        layout_score, sustainability_score, composite_score,
                        constraints_satisfied, violations, metadata,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (node_id) DO UPDATE SET
                        project_id = EXCLUDED.project_id,
                        parent_node_id = EXCLUDED.parent_node_id,
                        children_ids = EXCLUDED.children_ids,
                        node_type = EXCLUDED.node_type,
                        generation_level = EXCLUDED.generation_level,
                        iteration_number = EXCLUDED.iteration_number,
                        functions = EXCLUDED.functions,
                        behaviors = EXCLUDED.behaviors,
                        structures = EXCLUDED.structures,
                        layout = EXCLUDED.layout,
                        functional_score = EXCLUDED.functional_score,
                        behavioral_score = EXCLUDED.behavioral_score,
                        structural_score = EXCLUDED.structural_score,
                        layout_score = EXCLUDED.layout_score,
                        sustainability_score = EXCLUDED.sustainability_score,
                        composite_score = EXCLUDED.composite_score,
                        constraints_satisfied = EXCLUDED.constraints_satisfied,
                        violations = EXCLUDED.violations,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    node_dict['node_id'],
                    node_dict.get('project_id'),
                    node_dict.get('parent_node_id'),
                    Json(self._sanitize_data(node_dict.get('children_ids', []))),
                    node_dict['node_type'],
                    node_dict.get('generation_level', 0),
                    node_dict.get('iteration_number', 1),
                    Json(self._sanitize_data(node_dict.get('functions', {}))),
                    Json(self._sanitize_data(node_dict.get('behaviors', {}))),
                    Json(self._sanitize_data(node_dict.get('structures', {}))),
                    Json(self._sanitize_data(node_dict.get('layout'))),
                    node_dict.get('functional_score', 0.0),
                    node_dict.get('behavioral_score', 0.0),
                    node_dict.get('structural_score', 0.0),
                    node_dict.get('layout_score', 0.0),
                    node_dict.get('sustainability_score', 0.0),
                    node_dict.get('composite_score', 0.0),
                    Json(self._sanitize_data(node_dict.get('constraints_satisfied', {}))),
                    Json(self._sanitize_data(node_dict.get('violations', []))),
                    Json(self._sanitize_data(node_dict.get('metadata', {}))),
                    node_dict.get('created_at'),
                    node_dict.get('updated_at')
                ))
                self.conn.commit()
                logger.debug(f"✓ Stored FBSL node: {node_dict['node_id'][:8]}...")
                return True
        except Exception as e:
            logger.error(f"✗ Failed to store FBSL node: {e}")
            self.conn.rollback()
            return False
    
    def get_fbsl_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an FBSL node by ID"""
        try:
            result = self.execute_query(
                "SELECT * FROM fbsl_nodes WHERE node_id = %s",
                (node_id,)
            )
            if result:
                return dict(result[0])
            return None
        except Exception as e:
            logger.error(f"✗ Failed to retrieve node: {e}")
            return None
    
    def get_prototypes_by_project(self, project_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all prototypes for a project, sorted by composite score"""
        try:
            result = self.execute_query(
                """
                SELECT * FROM fbsl_nodes 
                WHERE project_id = %s AND node_type = 'design_prototype'
                ORDER BY composite_score DESC
                LIMIT %s
                """,
                (project_id, limit)
            )
            return [dict(row) for row in result] if result else []
        except Exception as e:
            logger.error(f"✗ Failed to retrieve prototypes: {e}")
            return []
    
    def store_project(self, project_id: str, project_name: str, 
                     description: str = None, requirements: str = None,
                     context: Dict = None) -> bool:
        """Store project information"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO projects (project_id, project_name, description, requirements, context)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (project_id) DO UPDATE SET
                        project_name = EXCLUDED.project_name,
                        description = EXCLUDED.description,
                        requirements = EXCLUDED.requirements,
                        context = EXCLUDED.context,
                        updated_at = CURRENT_TIMESTAMP
                """, (project_id, project_name, description, requirements, Json(context or {})))
                self.conn.commit()
                logger.debug(f"✓ Stored project: {project_id}")
                return True
        except Exception as e:
            logger.error(f"✗ Failed to store project: {e}")
            self.conn.rollback()
            return False
    
    def store_evaluation(self, evaluation_dict: Dict[str, Any]) -> bool:
        """Store detailed evaluation results"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO evaluations (
                        evaluation_id, node_id, project_id,
                        functional_adequacy, behavioral_performance,
                        structural_feasibility, layout_efficiency, sustainability,
                        composite_score, rank, strengths, weaknesses, recommendations
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (evaluation_id) DO UPDATE SET
                        functional_adequacy = EXCLUDED.functional_adequacy,
                        behavioral_performance = EXCLUDED.behavioral_performance,
                        structural_feasibility = EXCLUDED.structural_feasibility,
                        layout_efficiency = EXCLUDED.layout_efficiency,
                        sustainability = EXCLUDED.sustainability,
                        composite_score = EXCLUDED.composite_score,
                        rank = EXCLUDED.rank,
                        strengths = EXCLUDED.strengths,
                        weaknesses = EXCLUDED.weaknesses,
                        recommendations = EXCLUDED.recommendations
                """, (
                    evaluation_dict.get('evaluation_id'),
                    evaluation_dict.get('node_id'),
                    evaluation_dict.get('project_id'),
                    Json(evaluation_dict.get('functional_adequacy', {})),
                    Json(evaluation_dict.get('behavioral_performance', {})),
                    Json(evaluation_dict.get('structural_feasibility', {})),
                    Json(evaluation_dict.get('layout_efficiency', {})),
                    Json(evaluation_dict.get('sustainability', {})),
                    evaluation_dict.get('composite_score', 0.0),
                    evaluation_dict.get('rank', 0),
                    Json(evaluation_dict.get('strengths', [])),
                    Json(evaluation_dict.get('weaknesses', [])),
                    Json(evaluation_dict.get('recommendations', []))
                ))
                self.conn.commit()
                logger.debug(f"✓ Stored evaluation: {evaluation_dict.get('evaluation_id', 'unknown')}")
                return True
        except Exception as e:
            logger.error(f"✗ Failed to store evaluation: {e}")
            self.conn.rollback()
            return False
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            logger.info("✓ Database connection closed")