#!/usr/bin/env python
"""
Run End-to-End Inference Validation and generate INFERENCE_RUN_REPORT.md.
"""

from datetime import datetime
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

TEST_PORT = 8124
HOST = f"http://127.0.0.1:{TEST_PORT}"
CHECKPOINT_PATH = "artifacts/trainings/unified_orchestrator_checkpoint.pt"
REPORT_PATH = "artifacts/trainings/INFERENCE_RUN_REPORT.md"

def main():
    print(f"Starting E2E Inference Validation at {datetime.now().isoformat()}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Missing {CHECKPOINT_PATH}")
        return 1

    cmd = [
        "python",
        "-m",
        "src.api.inference_server",
        "--checkpoint",
        CHECKPOINT_PATH,
        "--port",
        str(TEST_PORT),
        "--device",
        "cpu"
    ]

    print(f"Launching inference server: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    ready = False
    for i in range(90):
        if process.poll() is not None:
            print(f"Server process crashed with exit code {process.returncode}!", flush=True)
            stdout, stderr = process.communicate()
            print("STDOUT:", stdout.decode())
            print("STDERR:", stderr.decode())
            sys.exit(1)

        try:
            req = urllib.request.Request(f"{HOST}/health")
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode())
                    if data.get("status") == "healthy":
                        ready = True
                        print("Server is healthy!", flush=True)
                        break
        except (TimeoutError, urllib.error.URLError):
            print(f"Waiting for server... ({i}/90)", flush=True)
        time.sleep(1.0)

    if not ready:
        print("Server failed to start in time.")
        process.terminate()
        return 1

    # Run journey tests
    state = [[[0.0]*7 for _ in range(6)] for _ in range(17)]
    payload = {
        "state": state,
        "use_hrm_decomposition": True,
        "use_mcts": True,
        "use_trm_refinement": True
    }

    print("Sending inference journey payload...")
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{HOST}/inference", data=data, headers={'Content-Type': 'application/json'})

    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            latency = time.time() - start_time
            print(f"Inference successful in {latency:.2f}s")
            success = True
    except urllib.error.HTTPError as e:
        print(f"Inference failed: {e}")
        res_data = {}
        success = False

    # Shutdown
    print("Shutting down server...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    stdout, stderr = process.communicate()
    print("SERVER STDERR:")
    print(stderr.decode())

    # Write report
    report = f"""# E2E Inference Run Report
    
**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Checkpoint**: `{CHECKPOINT_PATH}`
**Status**: {"✅ PASSED" if success else "❌ FAILED"}

## Journey Validation
- **Health Check**: ✅ PASSED
- **Inference Request**: {"✅ PASSED" if success else "❌ FAILED"}

### Response Metadata
```json
{json.dumps(res_data, indent=2)}
```
"""

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open("INFERENCE_RUN_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {REPORT_PATH}")
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
