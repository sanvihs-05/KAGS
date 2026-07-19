-- PostgreSQL Permission Setup Script
-- Run this as a PostgreSQL superuser (usually 'postgres')

-- Option 1: Grant CREATE privilege on public schema
GRANT CREATE ON SCHEMA public TO fbsl_user;

-- Option 2: Grant ALL privileges on public schema
-- GRANT ALL ON SCHEMA public TO fbsl_user;

-- Grant privileges on existing tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fbsl_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fbsl_user;

-- Grant privileges on future tables (for auto-created tables)
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT ALL ON TABLES TO fbsl_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT ALL ON SEQUENCES TO fbsl_user;

-- Verify permissions
SELECT 
    has_schema_privilege('fbsl_user', 'public', 'CREATE') as can_create,
    has_schema_privilege('fbsl_user', 'public', 'USAGE') as can_use;

