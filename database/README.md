# Database Setup Guide

## 🔧 Fixing Permission Errors

If you see `permission denied for schema public`, you need to grant privileges to your database user.

### Option 1: Grant Permissions (Recommended)

Connect to PostgreSQL as a superuser (usually `postgres`) and run:

```bash
psql -U postgres -d fbsl_kags
```

Then execute:
```sql
GRANT CREATE ON SCHEMA public TO fbsl_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fbsl_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fbsl_user;

-- For future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT ALL ON TABLES TO fbsl_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT ALL ON SEQUENCES TO fbsl_user;
```

Or use the provided script:
```bash
psql -U postgres -d fbsl_kags -f database/setup_permissions.sql
```

### Option 2: Create Tables Manually

If you can't grant permissions, create tables manually:

```bash
psql -U postgres -d fbsl_kags -f database/schema.sql
```

Then grant privileges on the created tables:
```sql
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fbsl_user;
```

### Option 3: Use a Different User

If you have admin access, you can use the `postgres` superuser directly in `config/config.yaml`:

```yaml
database:
  user: postgres
  password: your_postgres_password
```

---

## 📋 Quick Setup Commands

### 1. Create Database (if it doesn't exist)
```bash
psql -U postgres -c "CREATE DATABASE fbsl_kags;"
```

### 2. Create User (if needed)
```bash
psql -U postgres -c "CREATE USER fbsl_user WITH PASSWORD 'your_secure_password';"
```

### 3. Grant Permissions
```bash
psql -U postgres -d fbsl_kags -f database/setup_permissions.sql
```

### 4. Create Schema
```bash
psql -U postgres -d fbsl_kags -f database/schema.sql
```

---

## ✅ Verify Setup

Run the test script:
```bash
python backend/test_database.py
```

You should see:
- ✅ Database connection successful
- ✅ Schema creation successful (or tables already exist)
- ✅ Store and retrieve successful

---

## 🔍 Troubleshooting

### Error: "permission denied for schema public"
**Solution**: Run `database/setup_permissions.sql` as superuser

### Error: "relation does not exist"
**Solution**: Run `database/schema.sql` to create tables

### Error: "password authentication failed"
**Solution**: Check `config/config.yaml` has correct credentials

### Error: "could not connect to server"
**Solution**: 
- Ensure PostgreSQL is running: `pg_ctl status` or check services
- Verify host/port in config.yaml (default: localhost:5432)

---

## 📁 Files

- `schema.sql` - Database schema (tables, indexes)
- `setup_permissions.sql` - Permission grants
- `README.md` - This file

