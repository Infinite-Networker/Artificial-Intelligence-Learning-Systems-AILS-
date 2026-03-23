# Security Policy

**Artificial Intelligence Learning System (AILS)**
*Created by Cherry Computer Ltd.*

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | ✅ Active support  |
| < 1.0   | ❌ Not supported   |

## Reporting a Vulnerability

If you discover a security vulnerability in AILS, **please do not open a public GitHub issue.**

Instead, please report it privately:

**📧 Email**: security@cherrycomputer.ltd  
**Subject**: `[AILS SECURITY] Brief description`

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within **48 hours** and provide a resolution timeline.

## Security Best Practices

When using AILS:
- Never commit credentials, API keys, or database passwords
- Use environment variables for all sensitive configuration
- Keep dependencies updated with `pip install --upgrade -r requirements.txt`
- Run `safety check` to audit dependency vulnerabilities
- Ensure scrapers respect `robots.txt` and rate limits

*Cherry Computer Ltd. takes security seriously and will address all valid reports promptly.*
