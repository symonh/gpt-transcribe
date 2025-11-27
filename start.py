#!/usr/bin/env python3
"""
Single command to start the entire transcription app.
Runs Redis check, worker, and web server together.
"""
import subprocess
import sys
import os
import signal
import time
import atexit

processes = []

def cleanup():
    """Kill all child processes on exit."""
    for p in processes:
        if p.poll() is None:  # Still running
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

def check_redis():
    """Check if Redis is running, start it if not."""
    try:
        import redis
        r = redis.from_url('redis://localhost:6379')
        r.ping()
        print("✓ Redis is running")
        return True
    except Exception:
        print("✗ Redis not running. Starting Redis...")
        try:
            # Try to start Redis
            redis_proc = subprocess.Popen(
                ['redis-server'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            processes.append(redis_proc)
            time.sleep(1)
            # Verify it started
            import redis
            r = redis.from_url('redis://localhost:6379')
            r.ping()
            print("✓ Redis started")
            return True
        except FileNotFoundError:
            print("✗ Redis not installed. Install with: brew install redis")
            print("  Or run without Redis (sync mode - will block during transcription)")
            return False
        except Exception as e:
            print(f"✗ Failed to start Redis: {e}")
            return False

def main():
    # Register cleanup
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Use whatever Python is running this script
    python_cmd = sys.executable
    
    print("\n🎙️  GPT Transcribe - Starting up...\n")
    
    # Check/start Redis
    redis_available = check_redis()
    
    if redis_available:
        # Start the worker
        print("Starting worker...")
        worker_proc = subprocess.Popen(
            [python_cmd, 'worker.py'],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        processes.append(worker_proc)
        time.sleep(0.5)
        print("✓ Worker started")
    else:
        print("⚠ Running in sync mode (no background processing)")
    
    # Start the Flask app (foreground)
    print("Starting web server...")
    print("\n" + "="*50)
    print("🌐 App running at: http://localhost:5001")
    print("   Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    try:
        app_proc = subprocess.Popen(
            [python_cmd, 'app.py'],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        processes.append(app_proc)
        app_proc.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    cleanup()
    print("Goodbye!")

if __name__ == '__main__':
    main()

