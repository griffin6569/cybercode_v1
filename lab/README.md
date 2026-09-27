# CyberCodeMini Security Lab

Docker-based isolated environment for authorized security testing and evaluation.

## Overview

This lab provides deliberately vulnerable applications for testing CyberCodeMini's
ability to identify, explain, and remediate security vulnerabilities in a controlled,
isolated environment.

**All testing is conducted locally against localhost applications.**

## Architecture

```
CyberCodeMini Agent
        |
        v
  Agent Runtime
        |
        v
  Docker Sandbox
        |
        v
  localhost vulnerable app
```

## Vulnerable Applications (Planned)

| App | Vulnerabilities | Language |
|-----|----------------|----------|
| VulnFlask | SQL injection, XSS, path traversal | Python |
| AuthLab | Authentication bypass, weak sessions | Python |
| WebShop | SSRF, insecure deserialization | Node.js |

## Usage (Phase 13)

```bash
# Start the lab
docker-compose up -d

# Run a lab task
python -m agent.runtime --lab sqli_basic

# Stop the lab
docker-compose down
```

## Safety

- All containers run with no network access to external hosts
- Filesystem access is restricted to the container
- No cloud metadata endpoint access
- No host filesystem mounting
- All exercises are educational and authorized
