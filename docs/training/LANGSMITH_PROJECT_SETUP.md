# LangSmith Project Setup Guide
## For Training Program

**Purpose:** Configure LangSmith projects and datasets for the training program
**Audience:** Training coordinators, self-paced learners, instructors
**Time:** 15-30 minutes

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Create LangSmith Account](#create-langsmith-account)
3. [Create Training Projects](#create-training-projects)
4. [Configure Environment](#configure-environment)
5. [Verify Setup](#verify-setup)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- LangSmith account (free tier available)
- Python 3.11+ installed
- Repository cloned locally
- Internet connection

### Recommended
- Multiple projects for different training phases
- Separate projects for each cohort
- Team workspace (for collaborative training)

---

## Create LangSmith Account

### Step 1: Sign Up

1. **Go to LangSmith:**
   - URL: https://smith.langchain.com/
   - Click "Sign Up" or "Get Started"

2. **Create Account:**
   - Use work email for team features
   - Choose organization name (e.g., "YourCompany-Training")
   - Accept terms of service

3. **Verify Email:**
   - Check inbox for verification email
   - Click verification link

### Step 2: Get API Key

1. **Navigate to Settings:**
   - Click profile icon (top right)
   - Select "Settings" → "API Keys"

2. **Create API Key:**
   - Click "Create API Key"
   - Name: `training-program-key`
   - Permissions: Full access (for training purposes)
   - **Important:** Copy and save the key immediately (shown only once)

3. **Secure Your Key:**
   ```bash
   # Store in password manager or secure note
   # NEVER commit to git
   # Add to .gitignore if using .env file
   ```

---

## Create Training Projects

### Project Structure

We recommend creating separate projects for each training phase:

```
Organization: YourCompany-Training
├── training-2025-module-1      (Architecture)
├── training-2025-module-2      (Agents)
├── training-2025-module-3      (E2E Flows)
├── training-2025-module-4      (Tracing)
├── training-2025-module-5      (Experiments)
├── training-2025-module-6      (Python)
├── training-2025-module-7      (CI/CD)
└── training-2025-capstone      (Capstone Projects)
```

### Step 1: Create Project via UI

1. **Navigate to Projects:**
   - Click "Projects" in sidebar
   - Click "+ New Project"

2. **Project Settings:**
   - Name: `training-2025-module-1`
   - Description: "Module 1: System & Architecture Deep Dive"
   - Visibility: Private (team only) or Public (open training)

3. **Repeat for all modules** (or use script below)

### Step 2: Create Projects via Script

**File:** `scripts/setup_langsmith_projects.py`

```python
"""
Create LangSmith projects for training program.
"""
import os
from langsmith import Client

def setup_training_projects():
    """Create all training projects."""
    # Initialize client
    client = Client(
        api_key=os.getenv("LANGSMITH_API_KEY"),
        api_url="https://api.smith.langchain.com"
    )

    # Project definitions
    projects = [
        {
            "name": "training-2025-module-1",
            "description": "Module 1: System & Architecture Deep Dive",
            "metadata": {"module": 1, "topic": "architecture"}
        },
        {
            "name": "training-2025-module-2",
            "description": "Module 2: Agents Deep Dive (HRM, TRM, MCTS)",
            "metadata": {"module": 2, "topic": "agents"}
        },
        {
            "name": "training-2025-module-3",
            "description": "Module 3: E2E Flows & LangGraph Orchestration",
            "metadata": {"module": 3, "topic": "e2e_flows"}
        },
        {
            "name": "training-2025-module-4",
            "description": "Module 4: LangSmith Tracing Utilities & Patterns",
            "metadata": {"module": 4, "topic": "tracing"}
        },
        {
            "name": "training-2025-module-5",
            "description": "Module 5: Experiments & Datasets in LangSmith",
            "metadata": {"module": 5, "topic": "experiments"}
        },
        {
            "name": "training-2025-module-6",
            "description": "Module 6: 2025 Python Coding & Testing Practices",
            "metadata": {"module": 6, "topic": "python_practices"}
        },
        {
            "name": "training-2025-module-7",
            "description": "Module 7: CI/CD & Observability Integration",
            "metadata": {"module": 7, "topic": "cicd"}
        },
        {
            "name": "training-2025-capstone",
            "description": "Capstone Projects (Week 8)",
            "metadata": {"module": 8, "topic": "capstone"}
        }
    ]

    print("Creating LangSmith training projects...")

    for project_info in projects:
        try:
            # Create project
            project = client.create_project(
                project_name=project_info["name"],
                description=project_info["description"],
                metadata=project_info["metadata"]
            )
            print(f"✅ Created: {project_info['name']}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"⚠️  Exists: {project_info['name']}")
            else:
                print(f"❌ Error creating {project_info['name']}: {e}")

    print("\n✅ Setup complete! Projects ready for training.")
    print("\nNext steps:")
    print("1. Configure environment variables")
    print("2. Run verification script")
    print("3. Start Module 1!")

if __name__ == "__main__":
    # Check API key
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ Error: LANGSMITH_API_KEY not set")
        print("\nSet your API key:")
        print("  export LANGSMITH_API_KEY='your-key-here'  # Linux/Mac")
        print("  $env:LANGSMITH_API_KEY='your-key-here'   # Windows PowerShell")
        exit(1)

    setup_training_projects()
```

**Usage:**
```bash
# Set API key
export LANGSMITH_API_KEY="your-key-here"

# Run setup script
python scripts/setup_langsmith_projects.py
```

---

## Configure Environment

### Option 1: Environment Variables (Recommended)

**Linux/Mac (bash/zsh):**
```bash
# Add to ~/.bashrc or ~/.zshrc for persistence
export LANGSMITH_API_KEY="your-api-key-here"
export LANGSMITH_PROJECT="training-2025-module-1"
export LANGSMITH_TRACING_ENABLED="true"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"

# Reload shell config
source ~/.bashrc  # or ~/.zshrc
```

**Windows PowerShell:**
```powershell
# Add to PowerShell profile for persistence
$env:LANGSMITH_API_KEY="your-api-key-here"
$env:LANGSMITH_PROJECT="training-2025-module-1"
$env:LANGSMITH_TRACING_ENABLED="true"
$env:LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
```

**Verify:**
```bash
python -c "import os; print('API Key set:', bool(os.getenv('LANGSMITH_API_KEY')))"
```

### Option 2: `.env` File

**Create `.env` file:**
```bash
# Create .env in repository root
cat > .env << 'EOF'
# LangSmith Configuration for Training
LANGSMITH_API_KEY=your-api-key-here
LANGSMITH_PROJECT=training-2025-module-1
LANGSMITH_TRACING_ENABLED=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Optional: LLM Provider Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
EOF
```

**Add to `.gitignore`:**
```bash
echo ".env" >> .gitignore
```

**Load in Python:**
```python
from dotenv import load_dotenv
load_dotenv()

import os
print("LangSmith configured:", bool(os.getenv("LANGSMITH_API_KEY")))
```

### Option 3: Per-Module Configuration Script

**File:** `scripts/set_training_module.sh`

```bash
#!/bin/bash
# Set environment for specific training module

MODULE=$1

if [ -z "$MODULE" ]; then
    echo "Usage: ./scripts/set_training_module.sh <module_number>"
    echo "Example: ./scripts/set_training_module.sh 1"
    exit 1
fi

export LANGSMITH_PROJECT="training-2025-module-$MODULE"
export LANGSMITH_TRACING_ENABLED="true"

echo "✅ Environment set for Module $MODULE"
echo "Project: $LANGSMITH_PROJECT"
echo ""
echo "Run tests with:"
echo "  pytest tests/ -v"
```

**Usage:**
```bash
# Set module 1 environment
./scripts/set_training_module.sh 1

# Run module 1 tests
pytest tests/components/test_hrm_agent_traced.py -v
```

---

## Verify Setup

### Verification Script

**File:** `scripts/verification/verify_langsmith_setup.py`

```python
"""
Verify LangSmith setup for training program.
"""
import os
import sys
from langsmith import Client

def verify_environment():
    """Verify environment variables."""
    print("🔍 Checking environment variables...")

    required_vars = {
        "LANGSMITH_API_KEY": "API key for authentication",
        "LANGSMITH_PROJECT": "Current training project",
        "LANGSMITH_TRACING_ENABLED": "Enable tracing"
    }

    all_set = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {description}")
        else:
            print(f"  ❌ {var}: {description} - NOT SET")
            all_set = False

    return all_set

def verify_connection():
    """Verify connection to LangSmith API."""
    print("\n🔍 Checking LangSmith API connection...")

    try:
        client = Client()
        # Test API connection
        projects = list(client.list_projects(limit=1))
        print("  ✅ Connected to LangSmith API")
        return True
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False

def verify_projects():
    """Verify training projects exist."""
    print("\n🔍 Checking training projects...")

    try:
        client = Client()
        all_projects = list(client.list_projects())
        project_names = [p.name for p in all_projects]

        expected_projects = [
            "training-2025-module-1",
            "training-2025-module-2",
            "training-2025-module-3",
            "training-2025-module-4",
            "training-2025-module-5",
            "training-2025-module-6",
            "training-2025-module-7",
            "training-2025-capstone"
        ]

        found = 0
        for project in expected_projects:
            if project in project_names:
                print(f"  ✅ {project}")
                found += 1
            else:
                print(f"  ❌ {project} - NOT FOUND")

        print(f"\nFound {found}/{len(expected_projects)} training projects")
        return found == len(expected_projects)

    except Exception as e:
        print(f"  ❌ Error listing projects: {e}")
        return False

def verify_tracing():
    """Verify tracing works with a test run."""
    print("\n🔍 Testing trace creation...")

    try:
        from langchain.callbacks import tracing_v2_enabled

        with tracing_v2_enabled() as cb:
            # Simple test trace
            print("  Creating test trace...")

        print("  ✅ Tracing works! Check LangSmith UI for test trace.")
        return True

    except Exception as e:
        print(f"  ❌ Tracing failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("LangSmith Setup Verification")
    print("=" * 60)

    results = {
        "Environment": verify_environment(),
        "Connection": verify_connection(),
        "Projects": verify_projects(),
        "Tracing": verify_tracing()
    }

    print("\n" + "=" * 60)
    print("Verification Results")
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("1. Read docs/training/README.md")
        print("2. Start Module 1: docs/training/MODULE_1_ARCHITECTURE.md")
        print("3. Run first traced test: python scripts/smoke_test_traced.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Set LANGSMITH_API_KEY environment variable")
        print("- Run: python scripts/setup_langsmith_projects.py")
        print("- Check API key permissions in LangSmith UI")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python scripts/verification/verify_langsmith_setup.py
```

**Expected Output:**
```
============================================================
LangSmith Setup Verification
============================================================
🔍 Checking environment variables...
  ✅ LANGSMITH_API_KEY: API key for authentication
  ✅ LANGSMITH_PROJECT: Current training project
  ✅ LANGSMITH_TRACING_ENABLED: Enable tracing

🔍 Checking LangSmith API connection...
  ✅ Connected to LangSmith API

🔍 Checking training projects...
  ✅ training-2025-module-1
  ✅ training-2025-module-2
  ✅ training-2025-module-3
  ✅ training-2025-module-4
  ✅ training-2025-module-5
  ✅ training-2025-module-6
  ✅ training-2025-module-7
  ✅ training-2025-capstone

Found 8/8 training projects

🔍 Testing trace creation...
  Creating test trace...
  ✅ Tracing works! Check LangSmith UI for test trace.

============================================================
Verification Results
============================================================
Environment         : ✅ PASS
Connection          : ✅ PASS
Projects            : ✅ PASS
Tracing             : ✅ PASS
============================================================

🎉 All checks passed! You're ready to start training.

Next steps:
1. Read docs/training/README.md
2. Start Module 1: docs/training/MODULE_1_ARCHITECTURE.md
3. Run first traced test: python scripts/smoke_test_traced.py
```

---

## Troubleshooting

### Issue: "Invalid API Key"

**Symptoms:**
```
AuthenticationError: Invalid API key
```

**Solutions:**
1. **Verify key is set:**
   ```bash
   echo $LANGSMITH_API_KEY
   ```

2. **Check for typos:**
   - Copy key again from LangSmith UI
   - Ensure no extra spaces or quotes

3. **Regenerate key:**
   - Go to Settings → API Keys
   - Delete old key, create new one
   - Update environment variable

---

### Issue: "Project not found"

**Symptoms:**
```
Project 'training-2025-module-1' not found
```

**Solutions:**
1. **List existing projects:**
   ```python
   from langsmith import Client
   client = Client()
   for project in client.list_projects():
       print(project.name)
   ```

2. **Create missing project:**
   ```bash
   python scripts/setup_langsmith_projects.py
   ```

3. **Check project name:**
   - Ensure exact name match (case-sensitive)
   - Update `LANGSMITH_PROJECT` environment variable

---

### Issue: "Traces not appearing"

**Symptoms:**
- Tests run successfully
- No traces in LangSmith UI

**Solutions:**
1. **Enable tracing:**
   ```bash
   export LANGSMITH_TRACING_ENABLED=true
   ```

2. **Check project name:**
   ```bash
   echo $LANGSMITH_PROJECT
   ```

3. **Wait for propagation:**
   - Traces may take 1-2 minutes to appear
   - Refresh LangSmith UI

4. **Check firewall:**
   - Ensure `api.smith.langchain.com` is accessible
   - Test: `curl https://api.smith.langchain.com/health`

---

### Issue: "Rate limit exceeded"

**Symptoms:**
```
RateLimitError: Too many requests
```

**Solutions:**
1. **Upgrade plan:**
   - Free tier: 5,000 traces/month
   - Team tier: Higher limits

2. **Reduce trace volume:**
   - Only trace during development/debugging
   - Disable tracing for unit tests: `LANGSMITH_TRACING_ENABLED=false pytest tests/unit/`

3. **Use sampling:**
   ```python
   import random
   if random.random() < 0.1:  # 10% sampling
       with tracing_v2_enabled():
           # Traced code
   ```

---

## For Training Coordinators

### Multi-Cohort Setup

**Recommended Structure:**
```
Organization: YourCompany-Training
├── cohort-2025-q1/
│   ├── training-2025-q1-module-1
│   ├── training-2025-q1-module-2
│   └── ...
├── cohort-2025-q2/
│   ├── training-2025-q2-module-1
│   └── ...
```

**Benefits:**
- Isolate cohorts for clean comparison
- Track progress per cohort
- Retain historical data

### Team Access

**Invite Team Members:**
1. Go to Settings → Team
2. Click "Invite Member"
3. Enter email addresses
4. Set role: Viewer, Member, or Admin
5. Send invitations

**Roles:**
- **Viewer:** Read-only access (for stakeholders)
- **Member:** Full access to traces and datasets (for trainees)
- **Admin:** Full access + settings (for instructors)

---

## Quick Reference

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `LANGSMITH_API_KEY` | Authentication | `ls__...` |
| `LANGSMITH_PROJECT` | Current project | `training-2025-module-1` |
| `LANGSMITH_TRACING_ENABLED` | Enable/disable tracing | `true` or `false` |
| `LANGSMITH_ENDPOINT` | API endpoint | `https://api.smith.langchain.com` |

### Common Commands

```bash
# Setup
python scripts/setup_langsmith_projects.py

# Verify
python scripts/verification/verify_langsmith_setup.py

# Set module
./scripts/set_training_module.sh 1

# Run traced test
LANGSMITH_PROJECT=training-2025-module-1 pytest tests/components/ -v

# List projects
python -c "from langsmith import Client; [print(p.name) for p in Client().list_projects()]"
```

---

## Next Steps

After completing setup:

1. **Read Training Overview:**
   - [COMPREHENSIVE_TRAINING_PLAN.md](COMPREHENSIVE_TRAINING_PLAN.md)
   - [README.md](README.md)

2. **Start Module 1:**
   - [MODULE_1_ARCHITECTURE.md](MODULE_1_ARCHITECTURE.md)

3. **Run First Traced Test:**
   ```bash
   python scripts/smoke_test_traced.py
   ```

4. **Check LangSmith UI:**
   - https://smith.langchain.com/
   - View your first trace!

---

**Questions?**
- Review [TROUBLESHOOTING_PLAYBOOK.md](TROUBLESHOOTING_PLAYBOOK.md)
- Ask in training Slack/Discord
- Email: [training-lead@example.com]

**Happy Training! 🎉**
