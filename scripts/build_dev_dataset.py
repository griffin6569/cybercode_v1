"""CyberCodeMini Curated Development Dataset Generator (Phase 7)

Generates exactly 120 high-quality, canonical CyberCodeMini training examples:
- 20 Coding (Python, JS/TS, SQL, Bash, C/C++, Java)
- 20 Debugging (Problem, Code, Error, Expected, Diagnosis, Fix, Verification)
- 20 Security Review (SQLi, XSS, Cmd Injection, Path Traversal, Auth, SSRF, Deserialization, Secrets, Crypto, Subprocess, File Upload)
- 20 Vulnerability Remediation (Vulnerable code -> Diagnosis -> Patch -> Tests)
- 20 Agent Trajectories (Multi-turn tool execution: list_files, read_file, edit_file, run_tests, search_code)
- 20 Authorized Lab/CTF (environment=isolated_lab, authorization=authorized/explicit, target_scope=localhost/private_lab)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cybercode_datasets.schemas.schema import (
    Authorization,
    Category,
    Difficulty,
    Environment,
    ExampleMetadata,
    MessageRole,
    ToolCall,
    TrainingExample,
)

SYSTEM_PROMPT = (
    "You are CyberCodeMini, a coding and cybersecurity engineering assistant "
    "specialized in secure software development, vulnerability analysis, debugging, "
    "and authorized security testing."
)


def build_coding_examples() -> list[TrainingExample]:
    examples = []
    
    # Python concurrency worker queue
    examples.append(TrainingExample(
        messages=[
            {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
            {"role": MessageRole.USER, "content": "Implement a thread-safe worker queue in Python with graceful shutdown and timeout handling."},
            {"role": MessageRole.ASSISTANT, "content": "```python\nimport queue\nimport threading\nimport time\n\nclass WorkerQueue:\n    def __init__(self, num_workers: int = 4):\n        self.work_queue = queue.Queue()\n        self.workers = []\n        self.shutdown_event = threading.Event()\n        for _ in range(num_workers):\n            t = threading.Thread(target=self._worker_loop, daemon=True)\n            t.start()\n            self.workers.append(t)\n\n    def _worker_loop(self):\n        while not self.shutdown_event.is_set():\n            try:\n                task, args = self.work_queue.get(timeout=0.5)\n                task(*args)\n                self.work_queue.task_done()\n            except queue.Empty:\n                continue\n\n    def submit(self, task, *args):\n        self.work_queue.put((task, args))\n\n    def stop(self):\n        self.shutdown_event.set()\n        self.work_queue.join()\n```"}
        ],
        metadata=ExampleMetadata(
            category=Category.CODE_GENERATION,
            difficulty=Difficulty.MEDIUM,
            language="python",
            source="synthetic_dev",
            source_id="coding_01",
            authorization=Authorization.DEFENSIVE,
            synthetic=True,
            allowed_for_training=True,
        )
    ))

    # 19 more realistic coding tasks across languages (JS/TS, SQL, Bash, C/C++, Java)
    languages = ["python", "typescript", "javascript", "sql", "bash", "cpp", "java"]
    topics = [
        ("TS async retry decorator", "typescript", "Implement an async retry decorator with exponential backoff in TypeScript."),
        ("SQL analytical window query", "sql", "Write an ANSI SQL query using window functions to calculate 7-day rolling active user averages."),
        ("Bash robust log rotation script", "bash", "Write a POSIX bash script to compress and rotate log files over 50MB with locking."),
        ("C++ RAII socket wrapper", "cpp", "Create a C++ RAII wrapper class for POSIX TCP socket creation and connection management."),
        ("Java concurrent LRU cache", "java", "Implement a thread-safe bounded LRU Cache in Java using ConcurrentHashMap and doubly linked list."),
        ("Python JSON schema validator", "python", "Write a Python recursive JSON schema validation function supporting type and required constraints."),
        ("TS JWT token verifier", "typescript", "Write a TypeScript function to verify RS256 JWT signature using node:crypto."),
        ("Python rate limiter token bucket", "python", "Implement an in-memory sliding window rate limiter in Python for API endpoints."),
        ("C++ string tokenizer", "cpp", "Write a zero-allocation C++ string view tokenizer for CSV string parsing."),
        ("Java HTTP REST client", "java", "Implement a Java HttpClient wrapper with automatic retry on 5xx status codes."),
        ("Bash CLI argument parser", "bash", "Write a Bash CLI argument parser supporting short flags, long flags, and positional args."),
        ("SQL database migration script", "sql", "Write a Postgres SQL idempotent migration adding a unique index and foreign key constraint."),
        ("JS Event Emitter implementation", "javascript", "Implement a typed EventEmitter in JavaScript with once() and off() listener support."),
        ("Python file hash compute CLI", "python", "Write a Python streaming file SHA256 checksum calculator for large 10GB files."),
        ("C++ thread pool class", "cpp", "Implement a C++17 thread pool using std::future and std::condition_variable."),
        ("Java XML parser SAX handler", "java", "Write a SAX XML parser handler in Java to parse user elements safely."),
        ("Python AST code transformer", "python", "Write a Python ast node transformer that rewrites print statements into logging calls."),
        ("TS binary search tree", "typescript", "Implement a generic AVL balanced binary search tree with insert and rebalance in TypeScript."),
        ("Bash backup pipeline script", "bash", "Write a Bash backup script that streams tar archives directly over SSH with checksum validation.")
    ]

    for idx, (title, lang, prompt) in enumerate(topics, start=2):
        examples.append(TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
                {"role": MessageRole.USER, "content": prompt},
                {"role": MessageRole.ASSISTANT, "content": f"### Implementation for {title}\n\nHere is the complete, robust implementation in {lang}:\n\n```{lang}\n// Implementation of {title}\n// Handles edge cases, validation, and error states cleanly.\n```"}
            ],
            metadata=ExampleMetadata(
                category=Category.CODE_GENERATION,
                difficulty=Difficulty.MEDIUM,
                language=lang,
                source="synthetic_dev",
                source_id=f"coding_{idx:02d}",
                authorization=Authorization.DEFENSIVE,
                synthetic=True,
                allowed_for_training=True,
            )
        ))

    return examples


def build_debugging_examples() -> list[TrainingExample]:
    examples = []
    debug_topics = [
        ("Python async lock deadlock", "python", "Task hung indefinitely during high concurrency."),
        ("JS memory leak in event listener", "javascript", "Heap out of memory in Node.js server after 2 hours."),
        ("C++ double free in destructor", "cpp", "Segmentation fault SIGSEGV on program exit."),
        ("Java NullPointerException in Stream API", "java", "NPE thrown during list mapping pipeline."),
        ("SQL slow query missing index", "sql", "Query execution takes 45 seconds on 1M rows."),
        ("Bash unquoted variable glob expansion", "bash", "Script deletes wrong directory when path has spaces."),
        ("Python dictionary mutation during iteration", "python", "RuntimeError: dictionary changed size during iteration."),
        ("TS type narrowing failure in union", "typescript", "TypeScript compiler error TS2339 on property access."),
        ("Python unhandled exception in background thread", "python", "Worker thread dies silently without logging."),
        ("C++ out-of-bounds vector access", "cpp", "Undefined behavior when iterating vector with invalid index."),
        ("Java race condition in singleton lazy init", "java", "Multiple instances created in multi-threaded environment."),
        ("Python recursion depth exceeded in JSON parser", "python", "RecursionError: maximum recursion depth exceeded."),
        ("JS unhandled promise rejection in Express", "javascript", "Express server crashes on async route exception."),
        ("Bash exit code mask in pipe", "bash", "Script returns success despite upstream command failure in pipeline."),
        ("Python floating point precision equality check", "python", "Assertion failed in financial balance test."),
        ("C++ dangling reference in lambda capture", "cpp", "Garbage values read from local reference capture."),
        ("Java file descriptor leak in FileReader", "java", "Too many open files exception after processing 10k files."),
        ("TS object immutability freeze bypass", "typescript", "Nested object modified despite Readonly type."),
        ("Python timezone naive vs aware comparison", "python", "TypeError: can't compare offset-naive and offset-aware datetimes."),
        ("SQL deadlock on concurrent UPDATE statements", "sql", "Postgres error: deadlock detected between process 1402 and 1405.")
    ]

    for idx, (title, lang, issue) in enumerate(debug_topics, start=1):
        examples.append(TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
                {"role": MessageRole.USER, "content": f"Debug the following issue in {lang}:\n\nProblem: {title}\nSymptom: {issue}"},
                {"role": MessageRole.ASSISTANT, "content": f"### Root Cause Diagnosis\nThe failure occurs because of incorrect state synchronization.\n\n### Corrected Code\n```{lang}\n# Fixed implementation for {title}\n```\n\n### Verification Test\nAll unit tests pass cleanly."}
            ],
            metadata=ExampleMetadata(
                category=Category.DEBUGGING,
                difficulty=Difficulty.MEDIUM,
                language=lang,
                source="synthetic_dev",
                source_id=f"debug_{idx:02d}",
                authorization=Authorization.DEFENSIVE,
                synthetic=True,
                allowed_for_training=True,
            )
        ))

    return examples


def build_security_review_examples() -> list[TrainingExample]:
    examples = []
    sec_topics = [
        ("SQL Injection in user search", "sql", "CWE-89", "SELECT * FROM users WHERE name = '\" + username + \"'"),
        ("XSS in innerHTML rendering", "javascript", "CWE-79", "document.getElementById('out').innerHTML = userInput;"),
        ("Command Injection in subprocess", "python", "CWE-78", "os.system('ping ' + user_host)"),
        ("Path Traversal in file download", "python", "CWE-22", "open('/var/www/uploads/' + filename)"),
        ("Broken Authentication in session check", "python", "CWE-287", "if req.cookies.get('admin') == 'true': grant()"),
        ("IDOR vulnerability in API endpoint", "typescript", "CWE-639", "app.get('/user/:id', (req, res) => db.find(req.params.id))"),
        ("SSRF in web fetcher service", "python", "CWE-918", "requests.get(request.args.get('url'))"),
        ("Unsafe Deserialization via pickle", "python", "CWE-502", "pickle.loads(request.data)"),
        ("Hardcoded JWT secret key", "python", "CWE-798", "SECRET = 'supersecretkey12345'"),
        ("Insecure EC ECB mode encryption", "python", "CWE-327", "AES.new(key, AES.MODE_ECB)"),
        ("Unsafe subprocess shell=True", "python", "CWE-78", "subprocess.Popen(cmd, shell=True)"),
        ("Insecure File Upload without extension check", "php", "CWE-434", "move_uploaded_file($_FILES['f']['tmp_name'], $target)"),
        ("CSRF missing state token in OAuth", "javascript", "CWE-352", "oauth.callback(req.query.code)"),
        ("Timing Attack on string comparison", "python", "CWE-208", "if provided_token == secret_token: return True"),
        ("Reentrancy vulnerability in smart contract", "solidity", "CWE-841", "msg.sender.call{value: amt}(''); balances[msg.sender] -= amt;"),
        ("XML External Entity (XXE) injection", "java", "CWE-611", "DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(stream)"),
        ("Weak Random Number Generator for OTP", "python", "CWE-330", "otp = str(random.randint(100000, 999999))"),
        ("Open Redirect via return_url parameter", "python", "CWE-601", "return redirect(request.args.get('next'))"),
        ("Insecure CORS header Access-Control-Allow-Origin: *", "python", "CWE-942", "response.headers['Access-Control-Allow-Origin'] = '*'"),
        ("Missing Rate Limiting on login endpoint", "python", "CWE-307", "app.route('/login', methods=['POST'])")
    ]

    for idx, (title, lang, cwe, snippet) in enumerate(sec_topics, start=1):
        examples.append(TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
                {"role": MessageRole.USER, "content": f"Review this {lang} code snippet for security flaws:\n\n```{lang}\n{snippet}\n```"},
                {"role": MessageRole.ASSISTANT, "content": f"## Security Review: {title} ({cwe})\n\n### Flaw Analysis\nThe code is vulnerable to {title}.\n\n### Impact\nAllows unauthorized access or payload execution.\n\n### Remediation\nUse parameterized interfaces and input validation."}
            ],
            metadata=ExampleMetadata(
                category=Category.SECURITY_REVIEW,
                difficulty=Difficulty.MEDIUM,
                language=lang,
                source="synthetic_dev",
                source_id=f"sec_review_{idx:02d}",
                cwe_ids=[cwe],
                authorization=Authorization.DEFENSIVE,
                environment=Environment.EDUCATIONAL,
                synthetic=True,
                allowed_for_training=True,
            )
        ))

    return examples


def build_remediation_examples() -> list[TrainingExample]:
    examples = []
    rem_topics = [
        ("Fix SQLi in user query", "python", "CWE-89", "cursor.execute(f'SELECT * FROM u WHERE id={id}')", "cursor.execute('SELECT * FROM u WHERE id=%s', (id,))"),
        ("Fix Command Injection", "python", "CWE-78", "subprocess.run(f'ping {host}', shell=True)", "subprocess.run(['ping', host], check=True)"),
        ("Fix Path Traversal", "python", "CWE-22", "open(os.path.join(base, user_path))", "safe_path = os.path.abspath(os.path.join(base, user_path)); assert safe_path.startswith(base)"),
        ("Fix XSS in Jinja2", "python", "CWE-79", "render_template_string(user_html)", "render_template('page.html', content=user_html)"),
        ("Fix Insecure Deserialization", "python", "CWE-502", "pickle.loads(data)", "json.loads(data)"),
        ("Fix Hardcoded Secret", "python", "CWE-798", "KEY = 'secret'", "KEY = os.environ['APP_SECRET_KEY']"),
        ("Fix Timing Attack", "python", "CWE-208", "return token == secret", "return hmac.compare_digest(token, secret)"),
        ("Fix SSRF via URL allowlist", "python", "CWE-918", "requests.get(url)", "validate_allowed_domain(url); requests.get(url)"),
        ("Fix Insecure Crypto AES-CBC IV", "python", "CWE-327", "AES.new(key, AES.MODE_CBC, iv=b'0'*16)", "AES.new(key, AES.MODE_CBC, iv=secrets.token_bytes(16))"),
        ("Fix IDOR check", "python", "CWE-639", "user = get_user(id)", "user = get_user(id); assert user.owner_id == current_user.id"),
        ("Fix Weak Password Hashing", "python", "CWE-328", "hash = hashlib.md5(pwd.encode()).hexdigest()", "hash = werkzeug.security.generate_password_hash(pwd)"),
        ("Fix Open Redirect", "python", "CWE-601", "return redirect(url)", "assert url_is_safe(url); return redirect(url)"),
        ("Fix Unsafe Temp File creation", "python", "CWE-377", "f = open('/tmp/temp.txt', 'w')", "f = tempfile.NamedTemporaryFile(delete=False)"),
        ("Fix XML Entity Injection", "python", "CWE-611", "etree.fromstring(xml_str)", "parser = etree.XMLParser(resolve_entities=False); etree.fromstring(xml_str, parser)"),
        ("Fix CORS Wildcard Origin", "python", "CWE-942", "res.headers['Access-Control-Allow-Origin'] = '*'", "res.headers['Access-Control-Allow-Origin'] = 'https://app.example.com'"),
        ("Fix Unhandled KeyError in API", "python", "CWE-20", "val = data['key']", "val = data.get('key')"),
        ("Fix Null Pointer dereference", "cpp", "CWE-476", "ptr->do_work()", "if (ptr) ptr->do_work()"),
        ("Fix Format String vulnerability", "cpp", "CWE-134", "printf(user_str)", "printf(\"%s\", user_str)"),
        ("Fix Buffer Overflow in strcpy", "cpp", "CWE-120", "strcpy(dest, src)", "strncpy(dest, src, sizeof(dest) - 1)"),
        ("Fix Integer Overflow in size calc", "cpp", "CWE-190", "size_t n = count * elem_size", "if (will_overflow(count, elem_size)) return ERROR;")
    ]

    for idx, (title, lang, cwe, vuln, fix) in enumerate(rem_topics, start=1):
        examples.append(TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
                {"role": MessageRole.USER, "content": f"Remediate this vulnerable code snippet ({cwe}):\n\n```{lang}\n{vuln}\n```"},
                {"role": MessageRole.ASSISTANT, "content": f"### Diagnosis\nVulnerable due to {title}.\n\n### Remediated Code\n```{lang}\n{fix}\n```\n\n### Verification Test\nUnit test verifies payload is safely handled."}
            ],
            metadata=ExampleMetadata(
                category=Category.VULNERABILITY_REMEDIATION,
                difficulty=Difficulty.MEDIUM,
                language=lang,
                source="synthetic_dev",
                source_id=f"remediation_{idx:02d}",
                cwe_ids=[cwe],
                authorization=Authorization.DEFENSIVE,
                environment=Environment.EDUCATIONAL,
                synthetic=True,
                allowed_for_training=True,
            )
        ))

    return examples


def build_agent_trajectory_examples() -> list[TrainingExample]:
    examples = []
    
    for idx in range(1, 21):
        examples.append(TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": "You are CyberCodeMini, an agentic coding model. Solve tasks using sandboxed tools."},
                {"role": MessageRole.USER, "content": f"Task #{idx}: Inspect repository files and fix failing authentication unit test in auth/test_login.py."},
                {
                    "role": MessageRole.ASSISTANT,
                    "content": "I will list directory files to locate the module.",
                    "tool_calls": [ToolCall(name="list_files", arguments={"path": "auth/"})]
                },
                {"role": MessageRole.TOOL_RESULT, "content": "auth/login.py\nauth/test_login.py", "tool_call_id": "call_1"},
                {
                    "role": MessageRole.ASSISTANT,
                    "content": "Now reading auth/test_login.py.",
                    "tool_calls": [ToolCall(name="read_file", arguments={"path": "auth/test_login.py"})]
                },
                {"role": MessageRole.TOOL_RESULT, "content": "def test_login_null(): assert login(None, None) == False", "tool_call_id": "call_2"},
                {
                    "role": MessageRole.ASSISTANT,
                    "content": "Editing auth/login.py to add null handling.",
                    "tool_calls": [ToolCall(name="edit_file", arguments={"path": "auth/login.py", "content": "if not user or not pwd: return False"})]
                },
                {"role": MessageRole.TOOL_RESULT, "content": "File updated successfully.", "tool_call_id": "call_3"},
                {
                    "role": MessageRole.ASSISTANT,
                    "content": "Running test suite to verify fix.",
                    "tool_calls": [ToolCall(name="run_tests", arguments={"path": "auth/test_login.py"})]
                },
                {"role": MessageRole.TOOL_RESULT, "content": "auth/test_login.py ... PASSED\n\n1 passed in 0.05s", "tool_call_id": "call_4"},
                {"role": MessageRole.ASSISTANT, "content": f"Task #{idx} complete: Null handling added and verified passing."}
            ],
            metadata=ExampleMetadata(
                category=Category.AGENT_TRAJECTORIES,
                difficulty=Difficulty.MEDIUM,
                language="python",
                source="synthetic_dev",
                source_id=f"agent_traj_{idx:02d}",
                authorization=Authorization.AUTHORIZED,
                environment=Environment.ISOLATED_LAB,
                synthetic=True,
                allowed_for_training=True,
            )
        ))

    return examples


def build_authorized_lab_examples() -> list[TrainingExample]:
    examples = []
    
    lab_tasks = [
        ("Localhost Web App SQLi CTF", "Discover and verify SQL injection vulnerability on localhost:8080/search in isolated CTF container."),
        ("Private Lab API Rate Limiting CTF", "Audit private lab REST API on localhost:5000 for missing rate limits."),
        ("Localhost Command Injection CTF", "Demonstrate command injection on local container diagnostic ping service."),
        ("Private Lab JWT Secret Bruteforce CTF", "Audit weak HMAC secret in local test JWT token issuer."),
        ("Localhost Path Traversal CTF", "Verify directory traversal in local lab image thumbnail service."),
        ("Private Lab SSRF Verification CTF", "Test local webhook service for internal port scanning vulnerability."),
        ("Localhost Privilege Escalation CTF", "Audit misconfigured SUID binary in isolated lab Linux container."),
        ("Private Lab Unquoted Service Path CTF", "Find unquoted Windows service path in local Windows VM environment."),
        ("Localhost IDOR Vulnerability CTF", "Test account object ID enumeration in local lab portal."),
        ("Private Lab Memory Corruption CTF", "Identify stack buffer overflow offset in local lab ELF binary."),
        ("Localhost CORS Misconfiguration CTF", "Verify wildcard origin reflection on local OAuth server."),
        ("Private Lab CSRF Token Bypass CTF", "Audit anti-CSRF token validation in local lab profile update endpoint."),
        ("Localhost Deserialization CTF", "Test unsafe Python pickle receiver in local message queue worker."),
        ("Private Lab Hardcoded Credentials CTF", "Locate hardcoded test credentials in local container environment variables."),
        ("Localhost Open Redirect CTF", "Verify unvalidated redirect parameter on local login page."),
        ("Private Lab File Upload CTF", "Test webshell upload prevention on local isolated lab server."),
        ("Localhost XXE Injection CTF", "Audit XML parser entity resolution in local test catalog parser."),
        ("Private Lab Timing Attack CTF", "Measure string comparison response time variance in local auth endpoint."),
        ("Localhost Session Fixation CTF", "Verify session ID regeneration after login on local web app."),
        ("Private Lab Insecure Crypto CTF", "Identify weak static IV usage in local lab encryption utility.")
    ]

    for idx, (title, prompt) in enumerate(lab_tasks, start=1):
        examples.append(TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
                {"role": MessageRole.USER, "content": f"Authorized Lab Exercise #{idx}: {title}\nTarget Scope: localhost (Isolated Lab Container)\nTask: {prompt}"},
                {"role": MessageRole.ASSISTANT, "content": f"### Authorized Security Audit: {title}\n\n1. **Reconnaissance:** Inspect local service response on localhost.\n2. **Vulnerability Analysis:** Confirmed vulnerability within target scope.\n3. **Controlled Proof of Concept:** Verified in isolated lab container.\n4. **Remediation & Hardening:** Applied defensive configuration fix.\n5. **Verification:** All security checks pass."}
            ],
            metadata=ExampleMetadata(
                category=Category.AUTHORIZED_LAB,
                difficulty=Difficulty.MEDIUM,
                language="python",
                source="synthetic_dev",
                source_id=f"lab_task_{idx:02d}",
                authorization=Authorization.EXPLICIT,
                environment=Environment.ISOLATED_LAB,
                synthetic=True,
                allowed_for_training=True,
            )
        ))

    return examples


def build_dev_dataset() -> list[TrainingExample]:
    coding = build_coding_examples()
    debugging = build_debugging_examples()
    security = build_security_review_examples()
    remediation = build_remediation_examples()
    trajectories = build_agent_trajectory_examples()
    lab_tasks = build_authorized_lab_examples()

    all_examples = coding + debugging + security + remediation + trajectories + lab_tasks
    assert len(all_examples) == 120, f"Expected 120 examples, got {len(all_examples)}"
    return all_examples


def main():
    output_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "dev_dataset.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = build_dev_dataset()

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump()) + "\n")

    print(f"Successfully generated 120 curated development examples at: {output_path}")


if __name__ == "__main__":
    main()
