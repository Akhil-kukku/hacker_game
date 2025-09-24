"""
Self-Morphing AI Cybersecurity Engine - API Server
FastAPI server for the complete cybersecurity engine
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import json
import logging
import asyncio
from datetime import datetime
import threading
import time

from main_engine import SelfMorphingAICybersecurityEngine
from order_engine import NetworkFlow
from chaos_engine import AttackType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Self-Morphing AI Cybersecurity Engine API",
    description="API for the complete cybersecurity engine with ORDER, CHAOS, and BALANCE components",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = None
engine_thread = None

# Pydantic models for API requests/responses
class EngineConfig(BaseModel):
    simulation_mode: bool = True
    simulation_interval: float = 10.0
    batch_size: int = 100
    auto_optimization: bool = True

class NetworkFlowRequest(BaseModel):
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_count: int
    byte_count: int
    duration: float
    flags: str = ""

class AttackRequest(BaseModel):
    attack_type: str
    target_ip: str
    target_port: int = 80
    intensity: float = 1.0
    stealth_level: int = 5

class SystemStatus(BaseModel):
    system_running: bool
    simulation_mode: bool
    performance_metrics: Dict[str, Any]
    order_engine: Dict[str, Any]
    chaos_engine: Dict[str, Any]
    balance_controller: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize the engine on startup"""
    global engine, engine_thread
    
    try:
        logger.info("Initializing Self-Morphing AI Cybersecurity Engine...")
        engine = SelfMorphingAICybersecurityEngine()
        
        # Start engine in background thread
        def run_engine():
            try:
                engine.load_system_state()
                engine.start()
            except Exception as e:
                logger.error(f"Engine thread error: {e}")
        
        engine_thread = threading.Thread(target=run_engine, daemon=True)
        engine_thread.start()
        
        # Wait for engine to initialize
        await asyncio.sleep(2)
        logger.info("Engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the engine gracefully"""
    global engine
    if engine:
        logger.info("Shutting down engine...")
        engine.shutdown()

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "Self-Morphing AI Cybersecurity Engine",
        "version": "2.0.0",
        "status": "running",
        "components": ["ORDER", "CHAOS", "BALANCE"],
        "description": "Advanced cybersecurity engine with AI-powered defense, offense, and control systems"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if engine and engine.running:
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    else:
        raise HTTPException(status_code=503, detail="Engine not running")

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        status = engine.get_system_status()
        return SystemStatus(**status)
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
async def update_config(config: EngineConfig):
    """Update engine configuration"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Update configuration
        engine.config.update(config.dict())
        engine.simulation_mode = config.simulation_mode
        
        return {"message": "Configuration updated successfully", "config": config.dict()}
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flows")
async def process_network_flows(flows: List[NetworkFlowRequest]):
    """Process network flows through ORDER engine"""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    
    try:
        # Convert to NetworkFlow objects
        network_flows = []
        for flow_req in flows:
            flow = NetworkFlow(
                src_ip=flow_req.src_ip,
                dst_ip=flow_req.dst_ip,
                src_port=flow_req.src_port,
                dst_port=flow_req.dst_port,
                protocol=flow_req.protocol,
                packet_count=flow_req.packet_count,
                byte_count=flow_req.byte_count,
                duration=flow_req.duration,
                timestamp=time.time(),
                flags=flow_req.flags
            )
            network_flows.append(flow)
        
        # Process flows
        for flow in network_flows:
            engine.order_engine.process_flow(flow)
        
        return {
            "message": f"Processed {len(flows)} network flows",
            "flows_processed": len(flows),
            "order_status": engine.order_engine.get_status()
        }
    except Exception as e:
        logger.error(f"Failed to process flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attacks")
async def launch_attacks(attacks: List[AttackRequest]):
    """Launch attacks through CHAOS engine"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        attack_ids = []
        for attack_req in attacks:
            try:
                # Convert string to AttackType enum
                attack_type = AttackType(attack_req.attack_type)
                
                attack_id = engine.chaos_engine.launch_attack(
                    attack_type,
                    attack_req.target_ip,
                    attack_req.target_port
                )
                attack_ids.append(attack_id)
                
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid attack type: {attack_req.attack_type}")
        
        return {
            "message": f"Launched {len(attacks)} attacks",
            "attack_ids": attack_ids,
            "chaos_status": engine.chaos_engine.get_metrics()
        }
    except Exception as e:
        logger.error(f"Failed to launch attacks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order/status")
async def get_order_status():
    """Get ORDER engine status"""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    
    try:
        return engine.order_engine.get_status()
    except Exception as e:
        logger.error(f"Failed to get ORDER status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order/signatures")
async def get_attack_signatures(limit: int = 100):
    """Get attack signatures from ORDER engine"""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    
    try:
        return engine.order_engine.get_attack_signatures(limit)
    except Exception as e:
        logger.error(f"Failed to get attack signatures: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chaos/status")
async def get_chaos_status():
    """Get CHAOS engine status"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        return engine.chaos_engine.get_metrics()
    except Exception as e:
        logger.error(f"Failed to get CHAOS status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chaos/results")
async def get_attack_results(limit: int = 100):
    """Get attack results from CHAOS engine"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        return engine.chaos_engine.get_attack_results(limit)
    except Exception as e:
        logger.error(f"Failed to get attack results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chaos/patterns")
async def get_attack_patterns():
    """Get attack patterns from CHAOS engine"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        return engine.chaos_engine.get_attack_patterns()
    except Exception as e:
        logger.error(f"Failed to get attack patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chaos/aggression")
async def set_aggression_level(level: int):
    """Set CHAOS engine aggression level"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        if not 1 <= level <= 10:
            raise HTTPException(status_code=400, detail="Aggression level must be between 1 and 10")
        
        engine.chaos_engine.set_aggression_level(level)
        return {"message": f"Aggression level set to {level}"}
    except Exception as e:
        logger.error(f"Failed to set aggression level: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chaos/stealth")
async def set_stealth_mode(enabled: bool):
    """Set CHAOS engine stealth mode"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        engine.chaos_engine.set_stealth_mode(enabled)
        return {"message": f"Stealth mode {'enabled' if enabled else 'disabled'}"}
    except Exception as e:
        logger.error(f"Failed to set stealth mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance/status")
async def get_balance_status():
    """Get BALANCE controller status"""
    if not engine or not engine.balance_controller:
        raise HTTPException(status_code=503, detail="BALANCE controller not available")
    
    try:
        return engine.balance_controller.get_status()
    except Exception as e:
        logger.error(f"Failed to get BALANCE status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance/actions")
async def get_action_history(limit: int = 100):
    """Get BALANCE controller action history"""
    if not engine or not engine.balance_controller:
        raise HTTPException(status_code=503, detail="BALANCE controller not available")
    
    try:
        return engine.balance_controller.get_action_history(limit)
    except Exception as e:
        logger.error(f"Failed to get action history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance/rewards")
async def get_reward_history(limit: int = 100):
    """Get BALANCE controller reward history"""
    if not engine or not engine.balance_controller:
        raise HTTPException(status_code=503, detail="BALANCE controller not available")
    
    try:
        return engine.balance_controller.get_reward_history(limit)
    except Exception as e:
        logger.error(f"Failed to get reward history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulations")
async def get_simulation_results(limit: int = 100):
    """Get simulation results"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    
    try:
        return engine.get_simulation_results(limit)
    except Exception as e:
        logger.error(f"Failed to get simulation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tracking")
async def get_attack_response_tracking(limit: int = 100):
    """Get attack-response tracking data"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    
    try:
        return engine.get_attack_response_tracking(limit)
    except Exception as e:
        logger.error(f"Failed to get tracking data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New endpoints for enhanced cybersecurity capabilities

@app.get("/order/threat-indicators")
async def get_threat_indicators(limit: int = 100):
    """Get threat indicators from ORDER engine"""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    
    try:
        return engine.order_engine.get_threat_indicators(limit)
    except Exception as e:
        logger.error(f"Failed to get threat indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order/security-incidents")
async def get_security_incidents(limit: int = 50):
    """Get security incidents from ORDER engine"""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    
    try:
        return engine.order_engine.get_security_incidents(limit)
    except Exception as e:
        logger.error(f"Failed to get security incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/order/block-ip")
async def block_ip(ip: str, reason: str = "Manual block"):
    """Block an IP address"""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    
    try:
        engine.order_engine.block_ip(ip, reason)
        return {"message": f"IP {ip} blocked successfully", "reason": reason}
    except Exception as e:
        logger.error(f"Failed to block IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/order/unblock-ip")
async def unblock_ip(ip: str):
    """Unblock an IP address"""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    
    try:
        engine.order_engine.unblock_ip(ip)
        return {"message": f"IP {ip} unblocked successfully"}
    except Exception as e:
        logger.error(f"Failed to unblock IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chaos/intelligence-reports")
async def get_intelligence_reports(limit: int = 50):
    """Get intelligence reports from CHAOS engine"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        return engine.chaos_engine.get_intelligence_reports(limit)
    except Exception as e:
        logger.error(f"Failed to get intelligence reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chaos/backdoors")
async def get_backdoors_discovered(limit: int = 50):
    """Get discovered backdoors from CHAOS engine"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        return engine.chaos_engine.get_backdoors_discovered(limit)
    except Exception as e:
        logger.error(f"Failed to get backdoors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chaos/counterattacks")
async def get_counterattack_actions(limit: int = 50):
    """Get counterattack actions from CHAOS engine"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        return engine.chaos_engine.get_counterattack_actions(limit)
    except Exception as e:
        logger.error(f"Failed to get counterattack actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chaos/gather-intelligence")
async def gather_intelligence(target_ip: str):
    """Gather intelligence on a target IP"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        report = engine.chaos_engine.gather_intelligence(target_ip)
        return {"message": f"Intelligence gathering initiated for {target_ip}", "report": report.__dict__}
    except Exception as e:
        logger.error(f"Failed to gather intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chaos/hunt-backdoors")
async def hunt_backdoors(target_ip: str):
    """Hunt for backdoors on a target IP"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        backdoors = engine.chaos_engine.hunt_backdoors(target_ip)
        return {"message": f"Backdoor hunting completed for {target_ip}", "backdoors": [b.__dict__ for b in backdoors]}
    except Exception as e:
        logger.error(f"Failed to hunt backdoors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chaos/execute-counterattack")
async def execute_counterattack(target_ip: str, attack_type: str = "trace"):
    """Execute counterattack against a target IP"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        action = engine.chaos_engine.execute_counterattack(target_ip, attack_type)
        return {"message": f"Counterattack executed against {target_ip}", "action": action.__dict__}
    except Exception as e:
        logger.error(f"Failed to execute counterattack: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance/adaptation-events")
async def get_adaptation_events(limit: int = 50):
    """Get adaptation events from BALANCE controller"""
    if not engine or not engine.balance_controller:
        raise HTTPException(status_code=503, detail="BALANCE controller not available")
    
    try:
        return engine.balance_controller.get_adaptation_events(limit)
    except Exception as e:
        logger.error(f"Failed to get adaptation events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance/threat-patterns")
async def get_threat_patterns(limit: int = 50):
    """Get learned threat patterns from BALANCE controller"""
    if not engine or not engine.balance_controller:
        raise HTTPException(status_code=503, detail="BALANCE controller not available")
    
    try:
        return engine.balance_controller.get_threat_patterns(limit)
    except Exception as e:
        logger.error(f"Failed to get threat patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance/system-genomes")
async def get_system_genomes(limit: int = 20):
    """Get system genomes from BALANCE controller"""
    if not engine or not engine.balance_controller:
        raise HTTPException(status_code=503, detail="BALANCE controller not available")
    
    try:
        return engine.balance_controller.get_system_genomes(limit)
    except Exception as e:
        logger.error(f"Failed to get system genomes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/balance/predict-threat")
async def predict_threat(features: List[float]):
    """Predict threat type using AI models"""
    if not engine or not engine.balance_controller:
        raise HTTPException(status_code=503, detail="BALANCE controller not available")
    
    try:
        predictions = engine.balance_controller.predict_threat(features)
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Failed to predict threat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Training endpoints for developing base defense and attack capabilities

@app.post("/order/train")
async def train_order_engine(num_samples: int = 1000):
    """Train ORDER engine with known attack patterns"""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    
    try:
        # Generate training data
        training_data = engine.order_engine.generate_training_data(num_samples)
        
        # Train the model
        success = engine.order_engine.train_with_known_attacks(training_data)
        
        if success:
            # Evaluate performance
            performance = engine.order_engine.evaluate_model_performance()
            return {
                "message": "ORDER engine training completed successfully",
                "samples_used": len(training_data),
                "performance": performance
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")
    
    except Exception as e:
        logger.error(f"ORDER training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chaos/train")
async def train_chaos_engine(num_samples: int = 1000):
    """Train CHAOS engine with known attack patterns"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        # Generate training data
        training_data = engine.chaos_engine.generate_attack_training_data(num_samples)
        
        # Train the model
        success = engine.chaos_engine.train_attack_patterns(training_data)
        
        if success:
            # Evaluate performance
            performance = engine.chaos_engine.evaluate_attack_performance()
            return {
                "message": "CHAOS engine training completed successfully",
                "samples_used": len(training_data),
                "performance": performance
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")
    
    except Exception as e:
        logger.error(f"CHAOS training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/balance/train")
async def train_balance_controller(num_scenarios: int = 1000):
    """Train BALANCE controller with known scenarios"""
    if not engine or not engine.balance_controller:
        raise HTTPException(status_code=503, detail="BALANCE controller not available")
    
    try:
        # Generate training scenarios
        training_data = engine.balance_controller.generate_training_scenarios(num_scenarios)
        
        # Train the controller
        success = engine.balance_controller.train_with_known_scenarios(training_data)
        
        if success:
            # Evaluate performance
            performance = engine.balance_controller.evaluate_balance_performance()
            return {
                "message": "BALANCE controller training completed successfully",
                "scenarios_used": len(training_data),
                "performance": performance
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")
    
    except Exception as e:
        logger.error(f"BALANCE training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/all")
async def train_all_engines(order_samples: int = 1000, chaos_samples: int = 1000, balance_scenarios: int = 1000):
    """Train all engines with known attack patterns"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    
    try:
        results = {}
        
        # Train ORDER engine
        if engine.order_engine:
            training_data = engine.order_engine.generate_training_data(order_samples)
            order_success = engine.order_engine.train_with_known_attacks(training_data)
            results['order'] = {
                "success": order_success,
                "samples": len(training_data),
                "performance": engine.order_engine.evaluate_model_performance() if order_success else None
            }
        
        # Train CHAOS engine
        if engine.chaos_engine:
            training_data = engine.chaos_engine.generate_attack_training_data(chaos_samples)
            chaos_success = engine.chaos_engine.train_attack_patterns(training_data)
            results['chaos'] = {
                "success": chaos_success,
                "samples": len(training_data),
                "performance": engine.chaos_engine.evaluate_attack_performance() if chaos_success else None
            }
        
        # Train BALANCE controller
        if engine.balance_controller:
            training_data = engine.balance_controller.generate_training_scenarios(balance_scenarios)
            balance_success = engine.balance_controller.train_with_known_scenarios(training_data)
            results['balance'] = {
                "success": balance_success,
                "scenarios": len(training_data),
                "performance": engine.balance_controller.evaluate_balance_performance() if balance_success else None
            }
        
        return {
            "message": "All engines training completed",
            "results": results
        }
    
    except Exception as e:
        logger.error(f"All engines training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order/performance")
async def get_order_performance():
    """Get ORDER engine performance metrics"""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    
    try:
        performance = engine.order_engine.evaluate_model_performance()
        return performance
    except Exception as e:
        logger.error(f"Failed to get ORDER performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chaos/performance")
async def get_chaos_performance():
    """Get CHAOS engine performance metrics"""
    if not engine or not engine.chaos_engine:
        raise HTTPException(status_code=503, detail="CHAOS engine not available")
    
    try:
        performance = engine.chaos_engine.evaluate_attack_performance()
        return performance
    except Exception as e:
        logger.error(f"Failed to get CHAOS performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance/performance")
async def get_balance_performance():
    """Get BALANCE controller performance metrics"""
    if not engine or not engine.balance_controller:
        raise HTTPException(status_code=503, detail="BALANCE controller not available")
    
    try:
        performance = engine.balance_controller.evaluate_balance_performance()
        return performance
    except Exception as e:
        logger.error(f"Failed to get BALANCE performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def trigger_optimization():
    """Trigger system optimization"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    
    try:
        engine._optimize_system()
        return {"message": "System optimization triggered"}
    except Exception as e:
        logger.error(f"Failed to trigger optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save")
async def save_system_state():
    """Save current system state"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    
    try:
        engine.save_system_state()
        return {"message": "System state saved successfully"}
    except Exception as e:
        logger.error(f"Failed to save system state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load")
async def load_system_state():
    """Load previously saved system state"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    
    try:
        engine.load_system_state()
        return {"message": "System state loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load system state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time system updates"""
    await websocket.accept()
    
    try:
        while True:
            if engine and engine.running:
                # Send system status every 5 seconds
                status = engine.get_system_status()
                await websocket.send_text(json.dumps(status))
                await asyncio.sleep(5)
            else:
                await websocket.send_text(json.dumps({"error": "Engine not running"}))
                await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
