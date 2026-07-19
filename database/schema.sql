-- FBSL-KAGS Database Schema
-- Run this script as a PostgreSQL superuser or with appropriate privileges

-- FBSL Nodes table (prototypes, problems, evaluations)
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
    
    -- Foreign key constraint
    CONSTRAINT fk_parent FOREIGN KEY (parent_node_id) 
        REFERENCES fbsl_nodes(node_id) ON DELETE SET NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_node_type ON fbsl_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_project_id ON fbsl_nodes(project_id);
CREATE INDEX IF NOT EXISTS idx_composite_score ON fbsl_nodes(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_created_at ON fbsl_nodes(created_at DESC);

-- Projects table
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

-- Evaluations table (detailed evaluation results)
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

-- Grant privileges to application user (adjust username as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fbsl_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fbsl_user;

