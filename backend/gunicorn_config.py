# Gunicorn configuration for long-running video processing
bind = "0.0.0.0:8080"
workers = 1                    # Single worker prevents memory conflicts
worker_class = "sync"
timeout = 3600                 # 1 hour timeout
keepalive = 5
max_requests = 100
max_requests_jitter = 10
preload_app = True
graceful_timeout = 3600        # Graceful shutdown timeout
worker_tmp_dir = "/dev/shm"    # Use shared memory for better performance
