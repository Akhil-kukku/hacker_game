"""
Self-Morphing AI Cybersecurity Engine - Streamlit Dashboard
Interactive dashboard for monitoring and controlling the cybersecurity engine
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import asyncio
import threading
from typing import Dict, List, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Self-Morphing AI Cybersecurity Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

class DashboardManager:
    """Manages dashboard data and API interactions"""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.data_cache = {}
        self.last_update = None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status from API"""
        try:
            response = requests.get(f"{self.api_base_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def get_order_status(self) -> Dict[str, Any]:
        """Get ORDER engine status"""
        try:
            response = requests.get(f"{self.api_base_url}/order/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def get_chaos_status(self) -> Dict[str, Any]:
        """Get CHAOS engine status"""
        try:
            response = requests.get(f"{self.api_base_url}/chaos/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def get_balance_status(self) -> Dict[str, Any]:
        """Get BALANCE controller status"""
        try:
            response = requests.get(f"{self.api_base_url}/balance/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def get_simulation_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get simulation results"""
        try:
            response = requests.get(f"{self.api_base_url}/simulations?limit={limit}", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            return []
    
    def get_attack_signatures(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get attack signatures"""
        try:
            response = requests.get(f"{self.api_base_url}/order/signatures?limit={limit}", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            return []
    
    def get_attack_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get attack results"""
        try:
            response = requests.get(f"{self.api_base_url}/chaos/results?limit={limit}", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            return []
    
    def launch_attack(self, attack_type: str, target_ip: str, target_port: int = 80) -> Dict[str, Any]:
        """Launch an attack"""
        try:
            attack_data = [{
                "attack_type": attack_type,
                "target_ip": target_ip,
                "target_port": target_port
            }]
            response = requests.post(f"{self.api_base_url}/attacks", json=attack_data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def set_aggression_level(self, level: int) -> Dict[str, Any]:
        """Set CHAOS engine aggression level"""
        try:
            response = requests.post(f"{self.api_base_url}/chaos/aggression?level={level}", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def set_stealth_mode(self, enabled: bool) -> Dict[str, Any]:
        """Set CHAOS engine stealth mode"""
        try:
            response = requests.post(f"{self.api_base_url}/chaos/stealth?enabled={enabled}", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def trigger_optimization(self) -> Dict[str, Any]:
        """Trigger system optimization"""
        try:
            response = requests.post(f"{self.api_base_url}/optimize", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}

def create_gauge_chart(value: float, title: str, color: str = "blue") -> go.Figure:
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"},
                {'range': [0.7, 1], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_line_chart(data: List[Dict[str, Any]], x_key: str, y_key: str, title: str) -> go.Figure:
    """Create a line chart"""
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    df[x_key] = pd.to_datetime(df[x_key], unit='s')
    
    fig = px.line(df, x=x_key, y=y_key, title=title)
    fig.update_layout(height=400)
    return fig

def create_bar_chart(data: List[Dict[str, Any]], x_key: str, y_key: str, title: str) -> go.Figure:
    """Create a bar chart"""
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    fig = px.bar(df, x=x_key, y=y_key, title=title)
    fig.update_layout(height=400)
    return fig

def main():
    """Main dashboard function"""
    
    # Initialize dashboard manager
    dashboard = DashboardManager()
    
    # Header
    st.title("🛡️ Self-Morphing AI Cybersecurity Engine Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # System status indicator
    status = dashboard.get_system_status()
    if "error" not in status:
        st.sidebar.success("✅ System Connected")
    else:
        st.sidebar.error("❌ System Disconnected")
        st.sidebar.error(status["error"])
        return
    
    # Control buttons
    st.sidebar.subheader("System Controls")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    if st.sidebar.button("⚡ Trigger Optimization"):
        result = dashboard.trigger_optimization()
        if "error" not in result:
            st.sidebar.success("Optimization triggered!")
        else:
            st.sidebar.error(f"Error: {result['error']}")
    
    # CHAOS Engine Controls
    st.sidebar.subheader("🎯 CHAOS Engine Controls")
    
    aggression_level = st.sidebar.slider("Aggression Level", 1, 10, 5)
    if st.sidebar.button("Set Aggression"):
        result = dashboard.set_aggression_level(aggression_level)
        if "error" not in result:
            st.sidebar.success(f"Aggression set to {aggression_level}")
        else:
            st.sidebar.error(f"Error: {result['error']}")
    
    stealth_mode = st.sidebar.checkbox("Stealth Mode", value=True)
    if st.sidebar.button("Set Stealth"):
        result = dashboard.set_stealth_mode(stealth_mode)
        if "error" not in result:
            st.sidebar.success(f"Stealth mode {'enabled' if stealth_mode else 'disabled'}")
        else:
            st.sidebar.error(f"Error: {result['error']}")
    
    # Attack Launcher
    st.sidebar.subheader("🚀 Attack Launcher")
    
    attack_type = st.sidebar.selectbox(
        "Attack Type",
        ["DDoS", "Brute Force", "SQL Injection", "XSS", "Buffer Overflow", "Man in the Middle", "Phishing"]
    )
    
    target_ip = st.sidebar.text_input("Target IP", "192.168.1.1")
    target_port = st.sidebar.number_input("Target Port", 1, 65535, 80)
    
    if st.sidebar.button("Launch Attack"):
        result = dashboard.launch_attack(attack_type, target_ip, target_port)
        if "error" not in result:
            st.sidebar.success("Attack launched!")
        else:
            st.sidebar.error(f"Error: {result['error']}")
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    # System Overview
    with col1:
        st.subheader("📊 System Overview")
        
        # System balance gauge
        balance_score = status.get('performance_metrics', {}).get('system_balance_score', 0.0)
        balance_fig = create_gauge_chart(balance_score, "System Balance", "green")
        st.plotly_chart(balance_fig, use_container_width=True)
        
        # Key metrics
        metrics = status.get('performance_metrics', {})
        st.metric("Total Simulations", metrics.get('total_simulations', 0))
        st.metric("Successful Defenses", metrics.get('successful_defenses', 0))
        st.metric("Successful Attacks", metrics.get('successful_attacks', 0))
    
    # ORDER Engine Status
    with col2:
        st.subheader("🛡️ ORDER Engine (Defense)")
        
        order_status = dashboard.get_order_status()
        if "error" not in order_status:
            # Defense accuracy gauge
            accuracy = order_status.get('performance_metrics', {}).get('model_accuracy', 0.0)
            accuracy_fig = create_gauge_chart(accuracy, "Defense Accuracy", "blue")
            st.plotly_chart(accuracy_fig, use_container_width=True)
            
            # ORDER metrics
            perf_metrics = order_status.get('performance_metrics', {})
            st.metric("Flows Processed", perf_metrics.get('total_flows_processed', 0))
            st.metric("Anomalies Detected", perf_metrics.get('anomalies_detected', 0))
            st.metric("Model Mutations", order_status.get('mutation_counter', 0))
        else:
            st.error("ORDER Engine Error")
    
    # CHAOS Engine Status
    with col3:
        st.subheader("🎯 CHAOS Engine (Offense)")
        
        chaos_status = dashboard.get_chaos_status()
        if "error" not in chaos_status:
            # Attack success rate gauge
            success_rate = chaos_status.get('successful_attacks', 0) / max(chaos_status.get('total_attacks', 1), 1)
            success_fig = create_gauge_chart(success_rate, "Attack Success Rate", "red")
            st.plotly_chart(success_fig, use_container_width=True)
            
            # CHAOS metrics
            st.metric("Total Attacks", chaos_status.get('total_attacks', 0))
            st.metric("Successful Attacks", chaos_status.get('successful_attacks', 0))
            st.metric("Stealth Success Rate", f"{chaos_status.get('stealth_success_rate', 0.0):.2%}")
        else:
            st.error("CHAOS Engine Error")
    
    # BALANCE Controller Status
    st.subheader("⚖️ BALANCE Controller")
    
    balance_status = dashboard.get_balance_status()
    if "error" not in balance_status:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Actions", balance_status.get('metrics', {}).get('total_actions', 0))
        with col2:
            st.metric("Average Reward", f"{balance_status.get('average_reward', 0.0):.3f}")
        with col3:
            st.metric("Generation", balance_status.get('generation', 0))
        with col4:
            st.metric("Best Fitness", f"{balance_status.get('best_fitness', 0.0):.3f}")
        
        # Control weights
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Defense Weight", f"{balance_status.get('defense_weight', 0.5):.2f}")
        with col2:
            st.metric("Attack Weight", f"{balance_status.get('attack_weight', 0.5):.2f}")
    else:
        st.error("BALANCE Controller Error")
    
    # Charts Section
    st.subheader("📈 Performance Charts")
    
    # Get simulation data
    simulations = dashboard.get_simulation_results(50)
    if simulations:
        col1, col2 = st.columns(2)
        
        with col1:
            # System balance over time
            balance_data = [{'timestamp': s['timestamp'], 'balance': s['system_balance']} for s in simulations]
            balance_fig = create_line_chart(balance_data, 'timestamp', 'balance', 'System Balance Over Time')
            st.plotly_chart(balance_fig, use_container_width=True)
        
        with col2:
            # Attack vs Defense effectiveness
            attack_data = []
            defense_data = []
            for sim in simulations:
                attack_success = sim['attack_results']['successful_attacks'] / max(sim['attack_results']['total_attacks'], 1)
                defense_success = sim['defense_results']['anomalies_detected'] / max(sim['defense_results']['total_flows'], 1)
                
                attack_data.append({'timestamp': sim['timestamp'], 'effectiveness': attack_success})
                defense_data.append({'timestamp': sim['timestamp'], 'effectiveness': defense_success})
            
            # Create comparison chart
            if attack_data and defense_data:
                df_attack = pd.DataFrame(attack_data)
                df_defense = pd.DataFrame(defense_data)
                df_attack['timestamp'] = pd.to_datetime(df_attack['timestamp'], unit='s')
                df_defense['timestamp'] = pd.to_datetime(df_defense['timestamp'], unit='s')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_attack['timestamp'], y=df_attack['effectiveness'], 
                                       name='Attack Effectiveness', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df_defense['timestamp'], y=df_defense['effectiveness'], 
                                       name='Defense Effectiveness', line=dict(color='blue')))
                fig.update_layout(title='Attack vs Defense Effectiveness', height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Attack Signatures and Results
    st.subheader("🔍 Attack Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Attack Signatures")
        signatures = dashboard.get_attack_signatures(20)
        if signatures:
            df_sigs = pd.DataFrame(signatures)
            df_sigs['timestamp'] = pd.to_datetime(df_sigs['timestamp'], unit='s')
            st.dataframe(df_sigs[['name', 'category', 'confidence', 'timestamp']], use_container_width=True)
        else:
            st.info("No attack signatures available")
    
    with col2:
        st.subheader("🎯 Recent Attack Results")
        results = dashboard.get_attack_results(20)
        if results:
            df_results = pd.DataFrame(results)
            df_results['timestamp'] = pd.to_datetime(df_results['timestamp'], unit='s')
            st.dataframe(df_results[['attack_type', 'success', 'damage_dealt', 'timestamp']], use_container_width=True)
        else:
            st.info("No attack results available")
    
    # Footer
    st.markdown("---")
    st.markdown("🛡️ **Self-Morphing AI Cybersecurity Engine v2.0** | Built with Streamlit, FastAPI, and AI/ML")

if __name__ == "__main__":
    main()
