"""
FastAPI Dashboard Server for Real-Time GPU Profiling

This module provides a web dashboard with WebSocket support for real-time
profiling visualization. Designed to have zero overhead when not in use.
"""

import asyncio
import json
import time
from collections import deque
from typing import Dict, List, Optional, Set
from datetime import datetime
import threading
import pynvml

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    # Create dummy classes for type hints when not available
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = None
    HTTPException = None
    HTMLResponse = None
    FileResponse = None

from .error_handling import CUDANotAvailableError


class DashboardServer:
    """FastAPI server for real-time GPU profiling dashboard."""
    
    def __init__(self, port: int = 8080, max_history: int = 500):
        """
        Initialize dashboard server.
        
        Args:
            port: Port to run the server on
            max_history: Maximum number of profiling results to keep in memory
        """
        if not DASHBOARD_AVAILABLE:
            raise ImportError(
                "Dashboard dependencies not installed. Install with: "
                "pip install 'gpu-regime-profiler[dashboard]'"
            )
        
        self.port = port
        self.app = FastAPI(title="GPU Regime Profiler Dashboard")
        self.connected_clients: Set[WebSocket] = set()
        self.profiling_history: deque = deque(maxlen=max_history)
        self.gpu_metrics_history: deque = deque(maxlen=100)  # Keep last 100 GPU snapshots
        self.lock = threading.Lock()
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Setup routes
        self._setup_routes()
        
        # Setup GPU monitoring (will start on server startup)
        self._setup_gpu_monitoring()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve the dashboard HTML."""
            html_content = self._get_dashboard_html()
            return HTMLResponse(content=html_content)
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            # Send initial history
            try:
                await websocket.send_json({
                    "type": "history",
                    "data": list(self.profiling_history)
                })
                await websocket.send_json({
                    "type": "gpu_history",
                    "data": list(self.gpu_metrics_history)
                })
            except Exception:
                pass
            
            try:
                while True:
                    # Keep connection alive, wait for client messages
                    data = await websocket.receive_text()
                    # Echo back or handle client requests
                    if data == "ping":
                        await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                self.connected_clients.discard(websocket)
        
        @self.app.post("/api/profile")
        async def submit_profile(profile_data: dict):
            """Submit profiling data manually."""
            try:
                # Add timestamp if not present
                if "timestamp" not in profile_data:
                    profile_data["timestamp"] = time.time()
                
                # Store in history
                with self.lock:
                    self.profiling_history.append(profile_data)
                
                # Broadcast to connected clients
                await self._broadcast({
                    "type": "profile",
                    "data": profile_data
                })
                
                return {"status": "success", "message": "Profile data received"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/history")
        async def get_history():
            """Get recent profiling history."""
            with self.lock:
                return {"history": list(self.profiling_history)}
        
        @self.app.get("/api/gpu/status")
        async def get_gpu_status():
            """Get current GPU status."""
            try:
                gpu_status = self._get_gpu_metrics()
                return {"status": "success", "data": gpu_status}
            except Exception as e:
                return {"status": "error", "message": str(e)}
    
    async def _broadcast(self, message: dict):
        """Broadcast message to all connected WebSocket clients."""
        if not self.connected_clients:
            return
        
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected
    
    def add_profile(self, profile_data: dict):
        """
        Add profiling data (thread-safe, non-blocking).
        
        This method is called from the dashboard client.
        It's designed to be fast and non-blocking.
        """
        try:
            # Add timestamp
            if "timestamp" not in profile_data:
                profile_data["timestamp"] = time.time()
            
            # Store in history (thread-safe)
            with self.lock:
                self.profiling_history.append(profile_data)
            
            # Schedule async broadcast (non-blocking)
            if self.connected_clients:
                asyncio.create_task(self._broadcast({
                    "type": "profile",
                    "data": profile_data
                }))
        except Exception:
            # Silent failure - don't break profiling
            pass
    
    def _get_gpu_metrics(self) -> Dict:
        """Get current GPU metrics using pynvml."""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None
                
                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                except:
                    power = None
                
                gpu_data = {
                    "device_id": i,
                    "gpu_utilization": util.gpu,
                    "memory_utilization": util.memory,
                    "memory_used_mb": mem_info.used / (1024 ** 2),
                    "memory_total_mb": mem_info.total / (1024 ** 2),
                    "memory_free_mb": mem_info.free / (1024 ** 2),
                    "temperature": temp,
                    "power_watts": power,
                    "timestamp": time.time()
                }
                gpus.append(gpu_data)
            
            pynvml.nvmlShutdown()
            return {"gpus": gpus, "count": device_count}
        except Exception as e:
            return {"error": str(e), "gpus": [], "count": 0}
    
    def _setup_gpu_monitoring(self):
        """Setup GPU monitoring background task (starts on server startup)."""
        
        async def monitor_loop():
            """Background loop to monitor GPU metrics."""
            self.monitoring_active = True
            while self.monitoring_active:
                try:
                    gpu_metrics = self._get_gpu_metrics()
                    if "gpus" in gpu_metrics:
                        with self.lock:
                            self.gpu_metrics_history.append(gpu_metrics)
                        
                        # Broadcast to clients
                        await self._broadcast({
                            "type": "gpu_metrics",
                            "data": gpu_metrics
                        })
                except Exception:
                    pass  # Silent failure
                
                # Poll every 2-3 seconds (configurable)
                await asyncio.sleep(2.5)
        
        # Start monitoring task when server starts
        @self.app.on_event("startup")
        async def startup_event():
            """Start GPU monitoring on server startup."""
            try:
                self.monitoring_task = asyncio.create_task(monitor_loop())
            except Exception:
                pass
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Stop GPU monitoring on server shutdown."""
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
    
    def _get_dashboard_html(self) -> str:
        """Get the dashboard HTML content."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Regime Profiler Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            background: #f5f5f5;
            color: #333333;
            padding: 24px;
            line-height: 1.5;
        }
        .header {
            margin-bottom: 32px;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 24px;
        }
        .header h1 {
            color: #1a1a1a;
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 12px;
            letter-spacing: -0.5px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 24px;
            margin-bottom: 24px;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .card h2 {
            color: #1a1a1a;
            margin-bottom: 20px;
            font-size: 16px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .chart-container {
            position: relative;
            height: 280px;
        }
        .status {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            margin-bottom: 0;
            font-size: 13px;
        }
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-connected { background: #4caf50; }
        .status-disconnected { background: #f44336; }
        .status-text {
            color: #666666;
            font-size: 13px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { 
            color: #666666; 
            font-size: 13px;
            font-weight: 400;
        }
        .metric-value { 
            color: #1a1a1a; 
            font-weight: 500;
            font-size: 13px;
        }
        .regime-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .regime-OVERHEAD_BOUND { background: #ffebee; color: #c62828; }
        .regime-MEMORY_BOUND { background: #fff3e0; color: #e65100; }
        .regime-COMPUTE_BOUND { background: #e8f5e9; color: #2e7d32; }
    </style>
</head>
<body>
    <div class="header">
        <h1>GPU Regime Profiler Dashboard</h1>
        <div class="status" id="connectionStatus">
            <span>
                <span class="status-indicator status-disconnected" id="statusIndicator"></span>
                <span class="status-text" id="statusText">Disconnected</span>
            </span>
            <span class="status-text" id="lastUpdate">Never</span>
        </div>
    </div>
    
    <div class="grid">
        <div class="card">
            <h2>Runtime Over Time</h2>
            <div class="chart-container">
                <canvas id="runtimeChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>Regime Distribution</h2>
            <div class="chart-container">
                <canvas id="regimeChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>FLOPS Utilization</h2>
            <div class="chart-container">
                <canvas id="flopsChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>Memory Usage</h2>
            <div class="chart-container">
                <canvas id="memoryChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>GPU Utilization</h2>
            <div class="chart-container">
                <canvas id="gpuUtilChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>Latest Profile</h2>
            <div id="latestProfile">
                <p style="color: #999999; font-size: 13px;">Waiting for profile data...</p>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let ws = null;
        let charts = {};
        
        // Chart configurations - professional styling
        const chartConfigs = {
            runtime: {
                type: 'line',
                data: { labels: [], datasets: [{
                    label: 'Runtime',
                    data: [],
                    borderColor: '#1976d2',
                    backgroundColor: 'rgba(25, 118, 210, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBorderWidth: 2,
                    pointHoverBackgroundColor: '#1976d2',
                    pointHoverBorderColor: '#ffffff',
                    tension: 0.1,
                    fill: true
                }]},
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: { 
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12, weight: '600' },
                            bodyFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12 },
                            padding: 12,
                            cornerRadius: 4,
                            displayColors: false
                        }
                    },
                    scales: { 
                        y: { 
                            beginAtZero: true, 
                            ticks: { 
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 11 },
                                padding: 8
                            }, 
                            grid: { 
                                color: '#f0f0f0',
                                drawBorder: false
                            },
                            border: { display: false }
                        },
                        x: { 
                            ticks: { 
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 11 },
                                maxRotation: 0,
                                padding: 8
                            }, 
                            grid: { 
                                display: false,
                                drawBorder: false
                            },
                            border: { display: false }
                        } 
                    }
                }
            },
            regime: {
                type: 'doughnut',
                data: { labels: ['Overhead Bound', 'Memory Bound', 'Compute Bound'],
                        datasets: [{ data: [0, 0, 0],
                                   backgroundColor: ['#ef5350', '#ff9800', '#4caf50'],
                                   borderWidth: 0 }]},
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '70%',
                    plugins: { 
                        legend: { 
                            position: 'bottom',
                            labels: { 
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12 },
                                padding: 12,
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12, weight: '600' },
                            bodyFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12 },
                            padding: 12,
                            cornerRadius: 4
                        }
                    }
                }
            },
            flops: {
                type: 'line',
                data: { labels: [], datasets: [{
                    label: 'FLOPS Utilization',
                    data: [],
                    borderColor: '#7b1fa2',
                    backgroundColor: 'rgba(123, 31, 162, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBorderWidth: 2,
                    pointHoverBackgroundColor: '#7b1fa2',
                    pointHoverBorderColor: '#ffffff',
                    tension: 0.1,
                    fill: true
                }]},
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: { 
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12, weight: '600' },
                            bodyFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12 },
                            padding: 12,
                            cornerRadius: 4,
                            displayColors: false,
                            callbacks: {
                                label: function(context) {
                                    return context.parsed.y.toFixed(1) + '%';
                                }
                            }
                        }
                    },
                    scales: { 
                        y: { 
                            beginAtZero: true, 
                            max: 100,
                            ticks: { 
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 11 },
                                padding: 8,
                                callback: function(value) { return value + '%'; }
                            }, 
                            grid: { 
                                color: '#f0f0f0',
                                drawBorder: false
                            },
                            border: { display: false }
                        },
                        x: { 
                            ticks: { 
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 11 },
                                maxRotation: 0,
                                padding: 8
                            }, 
                            grid: { 
                                display: false,
                                drawBorder: false
                            },
                            border: { display: false }
                        } 
                    }
                }
            },
            memory: {
                type: 'line',
                data: { labels: [], datasets: [{
                    label: 'Memory',
                    data: [],
                    borderColor: '#f57c00',
                    backgroundColor: 'rgba(245, 124, 0, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBorderWidth: 2,
                    pointHoverBackgroundColor: '#f57c00',
                    pointHoverBorderColor: '#ffffff',
                    tension: 0.1,
                    fill: true
                }]},
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: { 
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12, weight: '600' },
                            bodyFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12 },
                            padding: 12,
                            cornerRadius: 4,
                            displayColors: false,
                            callbacks: {
                                label: function(context) {
                                    return context.parsed.y.toFixed(1) + ' MB';
                                }
                            }
                        }
                    },
                    scales: { 
                        y: { 
                            beginAtZero: true, 
                            ticks: { 
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 11 },
                                padding: 8
                            }, 
                            grid: { 
                                color: '#f0f0f0',
                                drawBorder: false
                            },
                            border: { display: false }
                        },
                        x: { 
                            ticks: { 
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 11 },
                                maxRotation: 0,
                                padding: 8
                            }, 
                            grid: { 
                                display: false,
                                drawBorder: false
                            },
                            border: { display: false }
                        } 
                    }
                }
            },
            gpuUtil: {
                type: 'line',
                data: { labels: [], datasets: []},
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: { 
                        legend: { 
                            position: 'bottom',
                            labels: {
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12 },
                                padding: 12,
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12, weight: '600' },
                            bodyFont: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 12 },
                            padding: 12,
                            cornerRadius: 4,
                            callbacks: {
                                label: function(context) {
                                    return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                                }
                            }
                        }
                    },
                    scales: { 
                        y: { 
                            beginAtZero: true, 
                            max: 100,
                            ticks: { 
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 11 },
                                padding: 8,
                                callback: function(value) { return value + '%'; }
                            }, 
                            grid: { 
                                color: '#f0f0f0',
                                drawBorder: false
                            },
                            border: { display: false }
                        },
                        x: { 
                            ticks: { 
                                color: '#666666',
                                font: { family: 'Helvetica Neue, Helvetica, Arial, sans-serif', size: 11 },
                                maxRotation: 0,
                                padding: 8
                            }, 
                            grid: { 
                                display: false,
                                drawBorder: false
                            },
                            border: { display: false }
                        } 
                    }
                }
            }
        };
        
        // Initialize charts
        function initCharts() {
            charts.runtime = new Chart(document.getElementById('runtimeChart'), chartConfigs.runtime);
            charts.regime = new Chart(document.getElementById('regimeChart'), chartConfigs.regime);
            charts.flops = new Chart(document.getElementById('flopsChart'), chartConfigs.flops);
            charts.memory = new Chart(document.getElementById('memoryChart'), chartConfigs.memory);
            charts.gpuUtil = new Chart(document.getElementById('gpuUtilChart'), chartConfigs.gpuUtil);
        }
        
        // Update charts with new data
        function updateCharts(profileData) {
            const timestamp = new Date(profileData.timestamp * 1000).toLocaleTimeString();
            
            // Runtime chart
            charts.runtime.data.labels.push(timestamp);
            charts.runtime.data.datasets[0].data.push(profileData.runtime_ms || 0);
            if (charts.runtime.data.labels.length > 50) {
                charts.runtime.data.labels.shift();
                charts.runtime.data.datasets[0].data.shift();
            }
            charts.runtime.update('none');
            
            // Regime distribution
            const regimeCounts = { OVERHEAD_BOUND: 0, MEMORY_BOUND: 0, COMPUTE_BOUND: 0 };
            charts.regime.data.datasets[0].data.forEach((val, idx) => {
                regimeCounts[charts.regime.data.labels[idx]] = val;
            });
            regimeCounts[profileData.regime || 'UNKNOWN'] = (regimeCounts[profileData.regime || 'UNKNOWN'] || 0) + 1;
            charts.regime.data.datasets[0].data = [
                regimeCounts.OVERHEAD_BOUND,
                regimeCounts.MEMORY_BOUND,
                regimeCounts.COMPUTE_BOUND
            ];
            charts.regime.update();
            
            // FLOPS utilization
            charts.flops.data.labels.push(timestamp);
            charts.flops.data.datasets[0].data.push((profileData.flops_utilization || 0) * 100);
            if (charts.flops.data.labels.length > 50) {
                charts.flops.data.labels.shift();
                charts.flops.data.datasets[0].data.shift();
            }
            charts.flops.update('none');
            
            // Memory chart
            charts.memory.data.labels.push(timestamp);
            const memPeak = profileData.memory?.peak_allocated_mb || 0;
            charts.memory.data.datasets[0].data.push(memPeak);
            if (charts.memory.data.labels.length > 50) {
                charts.memory.data.labels.shift();
                charts.memory.data.datasets[0].data.shift();
            }
            charts.memory.update('none');
            
            // Update latest profile display
            updateLatestProfile(profileData);
        }
        
        function updateGPUChart(gpuData) {
            if (!gpuData.gpus || gpuData.gpus.length === 0) return;
            
            const timestamp = new Date().toLocaleTimeString();
            
            // Update datasets for each GPU
            gpuData.gpus.forEach((gpu, idx) => {
                if (!charts.gpuUtil.data.datasets[idx]) {
                    const colors = ['#1976d2', '#7b1fa2', '#c2185b', '#f57c00', '#388e3c'];
                    charts.gpuUtil.data.datasets.push({
                        label: `GPU ${gpu.device_id}`,
                        data: [],
                        borderColor: colors[idx % colors.length],
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        pointHoverBorderWidth: 2,
                        tension: 0.1,
                        fill: false
                    });
                }
                charts.gpuUtil.data.datasets[idx].data.push(gpu.gpu_utilization);
                if (charts.gpuUtil.data.datasets[idx].data.length > 50) {
                    charts.gpuUtil.data.datasets[idx].data.shift();
                }
            });
            
            if (charts.gpuUtil.data.labels.length === 0 || 
                charts.gpuUtil.data.labels[charts.gpuUtil.data.labels.length - 1] !== timestamp) {
                charts.gpuUtil.data.labels.push(timestamp);
                if (charts.gpuUtil.data.labels.length > 50) {
                    charts.gpuUtil.data.labels.shift();
                }
            }
            charts.gpuUtil.update('none');
        }
        
        function updateLatestProfile(profile) {
            const html = `
                <div class="metric">
                    <span class="metric-label">Regime:</span>
                    <span class="regime-badge regime-${profile.regime || 'UNKNOWN'}">${profile.regime || 'UNKNOWN'}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Runtime:</span>
                    <span class="metric-value">${(profile.runtime_ms || 0).toFixed(2)} ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">FLOPS Utilization:</span>
                    <span class="metric-value">${((profile.flops_utilization || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Peak:</span>
                    <span class="metric-value">${(profile.memory?.peak_allocated_mb || 0).toFixed(1)} MB</span>
                </div>
                <div class="metric">
                    <span class="metric-label">OOM Risk:</span>
                    <span class="metric-value">${profile.memory?.oom_risk || 'N/A'}</span>
                </div>
            `;
            document.getElementById('latestProfile').innerHTML = html;
        }
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                document.getElementById('statusIndicator').className = 'status-indicator status-connected';
                document.getElementById('statusText').textContent = 'Connected';
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'profile') {
                    updateCharts(message.data);
                    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                } else if (message.type === 'gpu_metrics') {
                    updateGPUChart(message.data);
                } else if (message.type === 'history') {
                    message.data.forEach(updateCharts);
                } else if (message.type === 'gpu_history') {
                    message.data.forEach(updateGPUChart);
                }
            };
            
            ws.onerror = () => {
                document.getElementById('statusIndicator').className = 'status-indicator status-disconnected';
                document.getElementById('statusText').textContent = 'Error';
            };
            
            ws.onclose = () => {
                document.getElementById('statusIndicator').className = 'status-indicator status-disconnected';
                document.getElementById('statusText').textContent = 'Disconnected';
                setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
            };
        }
        
        // Initialize
        initCharts();
        connectWebSocket();
        
        // Keep connection alive
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 30000);
    </script>
</body>
</html>"""


def start_dashboard(port: int = 8080, host: str = "127.0.0.1"):
    """
    Start the dashboard server.
    
    Args:
        port: Port to run the server on (default: 8080)
        host: Host to bind to (default: 127.0.0.1)
    
    Returns:
        uvicorn.Server instance
    """
    if not DASHBOARD_AVAILABLE:
        raise ImportError(
            "Dashboard dependencies not installed. Install with: "
            "pip install 'gpu-regime-profiler[dashboard]'"
        )
    
    server = DashboardServer(port=port)
    config = uvicorn.Config(server.app, host=host, port=port, log_level="info")
    uvicorn_server = uvicorn.Server(config)
    
    print(f"GPU Regime Profiler Dashboard starting...")
    print(f"Dashboard available at: http://{host}:{port}")
    print(f"WebSocket endpoint: ws://{host}:{port}/ws")
    
    return uvicorn_server


def run_dashboard(port: int = 8080, host: str = "127.0.0.1"):
    """
    Run the dashboard server (blocking).
    
    Args:
        port: Port to run the server on (default: 8080)
        host: Host to bind to (default: 127.0.0.1)
    """
    server = start_dashboard(port=port, host=host)
    server.run()

