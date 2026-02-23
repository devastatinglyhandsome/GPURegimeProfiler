# Dashboard Security & Usability Fixes

## Summary
Fixed two major issues with the dashboard:
1. **Removed optional install** - Dashboard dependencies now included by default
2. **Removed ngrok entirely** - Security risk and doesn't work in prod network environments

## Changes Made

### 1. setup.py
- Moved dashboard dependencies from `extras_require['dashboard']` to `install_requires`
- Now includes: fastapi, uvicorn, websockets, requests, python-multipart by default
- Removed pyngrok dependency entirely
- Users can now just do: `pip install gpu-regime-profiler` (no more `[dashboard]` extras)

### 2. gpu_regime_profiler/dashboard.py
- **REMOVED**: All ngrok-related functions
  - `setup_ngrok_tunnel()`
  - `start_dashboard_with_ngrok()`
- **KEPT**: Local-only dashboard functions
  - `start_dashboard_server()` - now defaults to `host="127.0.0.1"` (local only)
  - `start_dashboard_colab()` - uses Colab's built-in port proxy
- **ADDED**: Documentation for SSH port forwarding as secure alternative

### 3. gpu_regime_profiler/__init__.py
- Removed exports: `start_dashboard_with_ngrok`, `setup_ngrok_tunnel`
- Kept exports: `start_dashboard_server`, `start_dashboard`, `start_dashboard_colab`

### 4. gpu_regime_profiler/profiler.py
- Removed `ngrok_token` class variable
- Removed `ngrok_token` parameter from `__init__()`
- Removed ngrok-related warning messages

### 5. example_dashboard.py
- Updated to use `start_dashboard_server()` instead of `start_dashboard_with_ngrok()`
- Simplified example - just local dashboard

### 6. README.md
- Completely rewrote dashboard section
- Removed all ngrok references
- Added SSH port forwarding section as secure alternative
- Updated install commands to remove `[dashboard]` extras
- Emphasized "Local access only - secure by default"

## Security Benefits

1. **No public URLs** - Dashboard only accessible locally or via SSH tunnel
2. **No third-party proxy** - Eliminates ngrok as attack vector
3. **No SIRT alerts** - Won't trigger "Unexpected proxy activity" detections
4. **Network policy compliant** - Doesn't create publicly accessible URLs to internal services

## Usage

### Local (same machine)
```python
from gpu_regime_profiler import start_dashboard_server
start_dashboard_server(port=8080)
# Open http://127.0.0.1:8080
```

### Remote (SSH port forwarding)
```bash
# On your laptop
ssh -L 8080:localhost:8080 user@remote-host

# Then open http://localhost:8080 in browser
```

### Google Colab
```python
from gpu_regime_profiler import start_dashboard_colab
start_dashboard_colab(port=8080)
# Uses Colab's built-in port preview
```

## Migration Guide

**Old code (with ngrok):**
```python
from gpu_regime_profiler import start_dashboard_with_ngrok
GPUProfiler.ngrok_token = "token"
url = start_dashboard_with_ngrok(port=8080)
```

**New code (local only):**
```python
from gpu_regime_profiler import start_dashboard_server
start_dashboard_server(port=8080)
# Open http://127.0.0.1:8080
```

**For remote access, use SSH:**
```bash
ssh -L 8080:localhost:8080 user@host
```

## Testing Checklist

- [ ] Install works without extras: `pip install gpu-regime-profiler`
- [ ] Dashboard starts locally: `start_dashboard_server(port=8080)`
- [ ] Dashboard accessible at http://127.0.0.1:8080
- [ ] No ngrok imports or references
- [ ] No SIRT alerts for proxy activity
- [ ] SSH port forwarding works for remote access
- [ ] Colab version works with built-in port preview
