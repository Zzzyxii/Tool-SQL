import socket
import subprocess

def get_host_ip():
    # Method 1: Try hostname -I first (most reliable on Linux clusters)
    try:
        result = subprocess.run("hostname -I", capture_output=True, text=True, shell=True).stdout
        ips = result.strip().split()
        # Filter for 172.x or 192.x or 10.x IPs if possible, or just take the first non-local one
        for ip in ips:
            if not ip.startswith("127."):
                return ip
    except:
        pass

    # Method 2: Fallback to socket connection (might fail without internet)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Use a private IP that is likely reachable or just a dummy one
        # We don't actually send data, just check routing table
        s.connect(("10.255.255.255", 1)) 
        ip = s.getsockname()[0]
        s.close()
        return ip.strip()
    except:
        pass
    
    # Method 3: Fallback to 8.8.8.8 (original method)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip.strip()
    except:
        return "127.0.0.1"

if __name__ == "__main__":
    print(get_host_ip())
