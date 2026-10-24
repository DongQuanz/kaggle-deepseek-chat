# Security review — 2026-10-24

This is a source and dependency review, not a penetration test or a guarantee that the application is vulnerability-free. Publishing source and operating an Internet-accessible service are separate risk decisions.

## Scope and findings

- Reviewed all files and the complete Git history available at review time (one initial commit), including the generated notebook. Exact comparisons against local credentials and common token/private-key patterns found no matches. Internal instructions, `.state/`, `.runs/`, run evidence, logs and weights were not tracked. Notebook outputs were empty. This is a scoped scan, not exhaustive secret detection.
- The local console previously relied on loopback binding without login. It now requires authentication in all launch modes. Its secret/run directories are explicitly blocked from Gradio file serving. CORS checks and HTML sanitization are explicit.
- On POSIX, secret directories/files are restricted to 0700/0600. Windows continues to depend on the user's filesystem ACLs; POSIX permission tests do not prove Windows isolation. Local administrators and processes running as the owner can read plaintext runtime credentials and the private key. `.gitignore` does not encrypt these files.
- Remote host credentials may only be sent to an HTTPS `*.gradio.live` origin, without embedded userinfo, a custom port, query, fragment or path. Cached origins are checked again before connecting. Automatic credential-bearing redirects and implicit Hugging Face credentials/file downloads are disabled for the remote client.
- Text payloads are validated and capped at 1 MiB/200 messages before model tokenization. This does not provide a network-level HTTP body limit or comprehensive denial-of-service protection. Authenticated guests can still consume shared GPU quota.
- Automatic inline password generation no longer prints its plaintext value to notebook logs. Retrieve it privately from the runtime environment or configure a Kaggle Secret.
- Shutdown requires both Gradio authentication and a separate admin token. Model requests remain text-only user/assistant messages with no execution tools or MCP server.

## Dependencies: unresolved findings

The initial `pip-audit -r requirements.txt` reported 39 advisory entries in `cryptography 49.0.0` and `Pillow 12.3.0` (the output includes duplicate cryptography entries). Cryptography has been raised to `>=50.0.2,<51` in both the local requirements and the Kaggle installer.

**Pillow advisories remain unresolved.** The scanner lists Pillow 13.0.0 as the fixed version, while Gradio 6.30.0 requires `pillow>=8.0,<13.0`. Forcing Pillow 13 with `--no-deps` would bypass the supported dependency contract and is not used here. The advisory classes include image/PDF parsing denial of service, memory safety issues, and code-execution scenarios. The application has no image/PDF model-input flow, but this is not proof that every framework path is unreachable.

The audit after the cryptography upgrade reports **37 advisory entries in Pillow 12.3.0 only**. A sanitized machine-readable list is in [security/dependency-findings.json](security/dependency-findings.json). Local HTTP checks confirmed anonymous console access returns 401, authenticated access returns 200, and attempts to serve `.state/access.json` return 403 without exposing its contents.

Do not treat this release as approved for an untrusted multi-tenant public service while these findings remain. A private, authenticated text-chat deployment has a smaller attack surface, with the dependency risk still present. Do not process untrusted image/PDF files in this environment. Re-run the audit after Gradio supports the patched Pillow release, or review a compatible upstream fix before upgrading.

References: [Gradio dependency metadata](https://pypi.org/pypi/gradio/6.30.0/json), [Pillow 13.0.0](https://pypi.org/project/pillow/13.0.0/), [cryptography 50.0.2](https://pypi.org/project/cryptography/50.0.2/).

## Reproduce

```bash
python -m pip install pip-audit
python -m pip_audit -r requirements.txt
python -m pytest -q tests -p no:cacheprovider
```

The dependency audit is expected to exit nonzero while the Pillow findings remain. Do not hide this with ignored advisory IDs. Dependabot is configured for weekly dependency update PRs; GitHub secret scanning and push protection were enabled on the repository at review time.

The review does not include a new Kaggle GPU run after these changes, a native llama.cpp/CUDA vulnerability audit, a browser penetration test, or verification of every dependency's exploitability. llama.cpp intentionally follows upstream master; model weights and source builds rely on upstream distribution integrity. Existing processes and Kaggle notebooks must be restarted/recreated to use new security code and dependencies.

## Reporting

Use GitHub's private vulnerability reporting feature if enabled. Do not post live passwords, keys, authenticated URLs, private notebook logs or exploit payloads containing personal data in public issues.
