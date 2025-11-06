"""
Generate a sample training dataset based on UNSW-NB15 schema
This creates a realistic synthetic dataset for initial training
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_normal_flows(n_samples=5000):
    """Generate normal network traffic patterns"""
    flows = []
    
    # Normal HTTP/HTTPS traffic
    for _ in range(n_samples // 3):
        flow = {
            'src_ip': f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            'dst_ip': f"8.8.{random.randint(1,254)}.{random.randint(1,254)}",
            'src_port': random.randint(49152, 65535),  # Ephemeral ports
            'dst_port': random.choice([80, 443]),
            'protocol': random.choice(['TCP', 'HTTP', 'HTTPS']),
            'packet_count': random.randint(10, 500),
            'byte_count': random.randint(500, 50000),
            'duration': random.uniform(0.1, 30.0),
            'flags': random.choice(['', 'ACK', 'PSH,ACK']),
            'label': 0,  # Normal
            'attack_cat': 'Normal'
        }
        flows.append(flow)
    
    # Normal DNS traffic
    for _ in range(n_samples // 3):
        flow = {
            'src_ip': f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            'dst_ip': f"8.8.{random.choice([4, 8])}.{random.choice([4, 8])}",
            'src_port': random.randint(49152, 65535),
            'dst_port': 53,
            'protocol': 'UDP',
            'packet_count': random.randint(2, 10),
            'byte_count': random.randint(64, 512),
            'duration': random.uniform(0.001, 1.0),
            'flags': '',
            'label': 0,  # Normal
            'attack_cat': 'Normal'
        }
        flows.append(flow)
    
    # Normal SSH/Email traffic
    for _ in range(n_samples // 3):
        flow = {
            'src_ip': f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            'dst_ip': f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'src_port': random.randint(49152, 65535),
            'dst_port': random.choice([22, 25, 587, 993]),
            'protocol': 'TCP',
            'packet_count': random.randint(20, 300),
            'byte_count': random.randint(1000, 20000),
            'duration': random.uniform(1.0, 60.0),
            'flags': random.choice(['ACK', 'PSH,ACK', 'FIN,ACK']),
            'label': 0,  # Normal
            'attack_cat': 'Normal'
        }
        flows.append(flow)
    
    return flows

def generate_ddos_attacks(n_samples=500):
    """Generate DDoS attack patterns"""
    flows = []
    
    for _ in range(n_samples):
        flow = {
            'src_ip': f"203.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'dst_ip': f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice([80, 443]),
            'protocol': random.choice(['TCP', 'UDP', 'ICMP']),
            'packet_count': random.randint(1000, 10000),  # High packet count
            'byte_count': random.randint(64000, 1000000),  # Large byte count
            'duration': random.uniform(0.01, 5.0),  # Short duration
            'flags': random.choice(['SYN', 'SYN,ACK', 'RST']),
            'label': 1,  # Attack
            'attack_cat': 'DoS'
        }
        flows.append(flow)
    
    return flows

def generate_port_scan_attacks(n_samples=300):
    """Generate port scanning attack patterns"""
    flows = []
    
    for _ in range(n_samples):
        flow = {
            'src_ip': f"203.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'dst_ip': f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'src_port': random.randint(40000, 50000),
            'dst_port': random.randint(1, 1024),  # Scanning well-known ports
            'protocol': 'TCP',
            'packet_count': random.randint(1, 10),  # Few packets
            'byte_count': random.randint(40, 200),  # Small size
            'duration': random.uniform(0.001, 0.1),  # Very short
            'flags': random.choice(['SYN', 'RST', 'SYN,RST']),
            'label': 1,  # Attack
            'attack_cat': 'Reconnaissance'
        }
        flows.append(flow)
    
    return flows

def generate_exploitation_attacks(n_samples=200):
    """Generate exploitation attack patterns (SQL injection, buffer overflow, etc.)"""
    flows = []
    
    for _ in range(n_samples):
        flow = {
            'src_ip': f"203.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'dst_ip': f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'src_port': random.randint(40000, 65535),
            'dst_port': random.choice([80, 443, 3306, 1433, 5432]),  # Web/DB ports
            'protocol': random.choice(['TCP', 'HTTP', 'HTTPS']),
            'packet_count': random.randint(50, 500),
            'byte_count': random.randint(5000, 100000),  # Larger payloads
            'duration': random.uniform(0.5, 10.0),
            'flags': random.choice(['PSH,ACK', 'ACK', 'FIN,ACK']),
            'label': 1,  # Attack
            'attack_cat': 'Exploits'
        }
        flows.append(flow)
    
    return flows

def generate_brute_force_attacks(n_samples=200):
    """Generate brute force attack patterns"""
    flows = []
    
    for _ in range(n_samples):
        flow = {
            'src_ip': f"203.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'dst_ip': f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'src_port': random.randint(40000, 65535),
            'dst_port': random.choice([22, 3389, 21, 23]),  # SSH, RDP, FTP, Telnet
            'protocol': 'TCP',
            'packet_count': random.randint(100, 1000),  # Many attempts
            'byte_count': random.randint(10000, 100000),
            'duration': random.uniform(10.0, 300.0),  # Long duration
            'flags': random.choice(['PSH,ACK', 'ACK']),
            'label': 1,  # Attack
            'attack_cat': 'Fuzzers'
        }
        flows.append(flow)
    
    return flows

def generate_backdoor_attacks(n_samples=100):
    """Generate backdoor/trojan patterns"""
    flows = []
    
    for _ in range(n_samples):
        flow = {
            'src_ip': f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",  # From inside
            'dst_ip': f"203.0.{random.randint(1,254)}.{random.randint(1,254)}",  # To outside
            'src_port': random.randint(40000, 65535),
            'dst_port': random.choice([4444, 6666, 8080, 31337]),  # Common backdoor ports
            'protocol': 'TCP',
            'packet_count': random.randint(50, 500),
            'byte_count': random.randint(5000, 50000),
            'duration': random.uniform(1.0, 60.0),
            'flags': random.choice(['PSH,ACK', 'ACK']),
            'label': 1,  # Attack
            'attack_cat': 'Backdoor'
        }
        flows.append(flow)
    
    return flows

def main():
    """Generate complete training dataset"""
    print("Generating synthetic training dataset...")
    
    # Generate different types of flows
    normal_flows = generate_normal_flows(10000)
    ddos_flows = generate_ddos_attacks(1000)
    scan_flows = generate_port_scan_attacks(500)
    exploit_flows = generate_exploitation_attacks(400)
    brute_flows = generate_brute_force_attacks(300)
    backdoor_flows = generate_backdoor_attacks(200)
    
    # Combine all flows
    all_flows = normal_flows + ddos_flows + scan_flows + exploit_flows + brute_flows + backdoor_flows
    
    # Shuffle
    random.shuffle(all_flows)
    
    # Create DataFrame
    df = pd.DataFrame(all_flows)
    
    # Add some additional features
    df['timestamp'] = [datetime.now() - timedelta(seconds=random.randint(0, 86400)) for _ in range(len(df))]
    df['flow_id'] = [f"flow_{i:06d}" for i in range(len(df))]
    
    # Save to CSV
    output_path = "../CSV Files/training_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset generated: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nAttack category distribution:")
    print(df['attack_cat'].value_counts())
    print(f"\nNormal traffic: {len(normal_flows)} ({len(normal_flows)/len(df)*100:.1f}%)")
    print(f"Attack traffic: {len(df) - len(normal_flows)} ({(len(df) - len(normal_flows))/len(df)*100:.1f}%)")
    
    # Create test set (20% split)
    test_size = int(len(df) * 0.2)
    test_df = df.sample(n=test_size, random_state=42)
    train_df = df.drop(test_df.index)
    
    train_df.to_csv("../CSV Files/training_data.csv", index=False)
    test_df.to_csv("../CSV Files/test_data.csv", index=False)
    
    print(f"\n📊 Train set: {len(train_df)} samples")
    print(f"📊 Test set: {len(test_df)} samples")

if __name__ == "__main__":
    main()
