# ImageWorks GUI - Background Service Management

## Quick Start

### Start GUI in Background
```bash
./scripts/start_gui_bg.sh
```

The GUI will run in the background and you can close your terminal. Access it at: **http://localhost:8501**

### Check Status
```bash
./scripts/status_gui.sh
```

Shows:
- Running status (PID, CPU, memory, uptime)
- Port listening status
- Access URL
- Helpful commands

### View Logs
```bash
tail -f logs/gui.log
```

Real-time log viewing. Press `Ctrl+C` to stop viewing (doesn't stop the GUI).

### Stop GUI
```bash
./scripts/stop_gui_bg.sh
```

Gracefully stops the background GUI process.

---

## Alternative: Foreground Mode

If you want to run the GUI in the foreground (blocks terminal but easier to stop with Ctrl+C):

```bash
./scripts/launch_gui.sh
```

Press `Ctrl+C` to stop.

---

## Systemd Service (Optional)

For automatic startup on boot:

### 1. Install Service

```bash
# Edit the service file with your username and paths
sed -i "s/%USER%/$USER/g" _staging/imageworks-gui.service
sed -i "s|%WORKDIR%|$PWD|g" _staging/imageworks-gui.service
sed -i "s|%VENV%|$PWD/.venv|g" _staging/imageworks-gui.service

# Copy to systemd
sudo cp _staging/imageworks-gui.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload
```

### 2. Enable and Start

```bash
# Start now
sudo systemctl start imageworks-gui

# Enable on boot
sudo systemctl enable imageworks-gui

# Check status
sudo systemctl status imageworks-gui
```

### 3. Control Service

```bash
# Stop
sudo systemctl stop imageworks-gui

# Restart
sudo systemctl restart imageworks-gui

# View logs
sudo journalctl -u imageworks-gui -f
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find what's using port 8501
sudo lsof -i :8501

# Or
ss -tuln | grep 8501

# Kill old process
pkill -f "streamlit run"
```

### GUI Won't Start

Check logs:
```bash
cat logs/gui.log
```

Common issues:
- **Missing dependencies**: Run `uv sync`
- **Python environment**: Check `.venv` exists
- **Port conflict**: Another app using 8501

### "Operation not supported" Error

This is fixed by the Streamlit config file at `~/.streamlit/config.toml` which disables browser auto-open.

If you still see it:
```bash
# Verify config exists
cat ~/.streamlit/config.toml

# Should show:
# [browser]
# gatherUsageStats = false
# serverAddress = "localhost"
#
# [server]
# headless = true
```

---

## Comparison: Background vs Foreground

| Feature | Background (`start_gui_bg.sh`) | Foreground (`launch_gui.sh`) |
|---------|-------------------------------|------------------------------|
| Terminal freed | ✅ Yes | ❌ No (blocks terminal) |
| Easy to stop | Run stop script | `Ctrl+C` |
| Auto-restart on crash | ❌ No (use systemd) | ❌ No |
| Logs | In `logs/gui.log` | In terminal |
| Best for | Always-on server | Development/testing |

---

## Usage Tips

### 1. **Starting on Boot**
Use the systemd service method above for automatic startup.

### 2. **Remote Access**
To access from other machines on your network:

Edit launch script to bind to all interfaces:
```bash
--server.address=0.0.0.0
```

Then access via: `http://YOUR_IP:8501`

⚠️ **Security Warning**: Only do this on trusted networks!

### 3. **Change Port**
Edit the scripts and replace `8501` with your preferred port:
```bash
--server.port=8502
```

### 4. **Multiple Instances**
To run multiple GUIs (different projects):
- Change the port for each instance
- Update the `pgrep` patterns in stop/status scripts

### 5. **Memory Management**
If the GUI uses too much memory:
```bash
# Restart periodically
./scripts/stop_gui_bg.sh && ./scripts/start_gui_bg.sh

# Or set up a cron job
# Add to crontab: 0 4 * * * /path/to/scripts/stop_gui_bg.sh && /path/to/scripts/start_gui_bg.sh
```

---

## Files Created

- **`scripts/start_gui_bg.sh`**: Start in background
- **`scripts/stop_gui_bg.sh`**: Stop background process
- **`scripts/status_gui.sh`**: Check status
- **`scripts/launch_gui.sh`**: Start in foreground (updated)
- **`_staging/imageworks-gui.service`**: Systemd service template
- **`~/.streamlit/config.toml`**: Streamlit configuration

---

## Quick Reference

```bash
# Start
./scripts/start_gui_bg.sh

# Check
./scripts/status_gui.sh

# Logs
tail -f logs/gui.log

# Stop
./scripts/stop_gui_bg.sh

# Restart
./scripts/stop_gui_bg.sh && ./scripts/start_gui_bg.sh

# Access
open http://localhost:8501
```

---

**Now your GUI runs in the background and doesn't hog the terminal!** 🎉
