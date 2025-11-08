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
import traceback

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

class DatasetTrainRequest(BaseModel):
    file_path: str
    label_column: str | None = None

class FeedbackRequest(BaseModel):
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_count: int
    byte_count: int
    duration: float
    flags: str = ""
    is_attack: bool

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
        
        # NEW: Check for training datasets and train ORDER engine
        import os
        dataset_candidates = [
            "CSV Files/training_data.csv",
            "../CSV Files/training_data.csv",
            "CSV Files/UNSW_NB15_1.csv",
            "../CSV Files/UNSW_NB15_1.csv",
            "CSV Files/CICIDS2017_sample.csv"
        ]
        
        dataset_found = False
        for dataset_path in dataset_candidates:
            if os.path.exists(dataset_path):
                logger.info(f"🎓 Found training dataset: {dataset_path}")
                try:
                    # Train ORDER engine with labeled data
                    logger.info("Training ORDER engine on real dataset...")
                    engine.order_engine.train_from_dataset(
                        dataset_path, 
                        label_column='label'  # 0=normal, 1=attack
                    )
                    logger.info("✅ ORDER engine trained successfully on real data!")
                    logger.info(f"Model status: {engine.order_engine.get_status()}")
                    dataset_found = True
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Training failed on {dataset_path}: {e}")
                    logger.info("Trying next dataset...")
        
        if not dataset_found:
            logger.warning("⚠️ No training datasets found - ORDER will use online learning only")
            logger.info("To add training data, place CSV files in 'CSV Files/' directory")
            logger.info("Expected columns: src_ip, dst_ip, src_port, dst_port, protocol, packet_count, byte_count, duration, flags, label")
        
        # If test dataset exists, run an initial evaluation to set baseline
        test_path_chosen = None
        if dataset_found:
            import os
            test_candidates = [
                "CSV Files/test_data.csv",
                "../CSV Files/test_data.csv"
            ]
            for test_path in test_candidates:
                if os.path.exists(test_path):
                    try:
                        logger.info(f"📊 Evaluating ORDER engine on test dataset: {test_path}")
                        metrics = engine.order_engine.evaluate_dataset(test_path, label_column='label')
                        logger.info(f"Evaluation metrics: {metrics}")
                        test_path_chosen = test_path
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Evaluation failed on {test_path}: {e}")

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

        # Schedule periodic evaluation to measure improvements over time
        if test_path_chosen:
            def periodic_eval():
                while True:
                    try:
                        time.sleep(15 * 60)  # every 15 minutes
                        if engine and engine.order_engine:
                            m = engine.order_engine.evaluate_dataset(test_path_chosen, label_column='label')
                            logger.info(f"Periodic evaluation metrics: {m}")
                    except Exception as ex:
                        logger.warning(f"Periodic evaluation failed: {ex}")
                        time.sleep(60)
            th = threading.Thread(target=periodic_eval, daemon=True)
            th.start()
        
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

@app.post("/order/train-dataset")
async def order_train_dataset(req: DatasetTrainRequest):
    """Train ORDER engine from a dataset (CSV/Parquet)."""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    try:
        engine.order_engine.train_from_dataset(req.file_path, req.label_column)
        return {"message": "ORDER trained from dataset", "order_status": engine.order_engine.get_status()}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Dataset training failed: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"{e}\n{tb}")

@app.post("/order/evaluate-dataset")
async def order_evaluate_dataset(req: DatasetTrainRequest):
    """Evaluate ORDER engine on a labeled dataset (CSV/Parquet)."""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    try:
        metrics = engine.order_engine.evaluate_dataset(req.file_path, req.label_column)
        return {"message": "Evaluation completed", "metrics": metrics, "summary": engine.order_engine.performance_metrics.get('evaluation_summary', {})}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Dataset evaluation failed: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"{e}\n{tb}")

@app.get("/order/evaluation-history")
async def order_evaluation_history(limit: int = 50):
    """Get recent evaluation history entries."""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    try:
        return {
            "history": engine.order_engine.get_evaluation_history(limit),
            "summary": engine.order_engine.performance_metrics.get('evaluation_summary', {})
        }
    except Exception as e:
        logger.error(f"Failed to get evaluation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/order/feedback")
async def order_feedback(req: FeedbackRequest):
    """Submit a labeled flow for online learning feedback."""
    if not engine or not engine.order_engine:
        raise HTTPException(status_code=503, detail="ORDER engine not available")
    try:
        flow = NetworkFlow(
            src_ip=req.src_ip,
            dst_ip=req.dst_ip,
            src_port=req.src_port,
            dst_port=req.dst_port,
            protocol=req.protocol,
            packet_count=req.packet_count,
            byte_count=req.byte_count,
            duration=req.duration,
            timestamp=time.time(),
            flags=req.flags
        )
        engine.order_engine.submit_feedback(flow, is_attack=req.is_attack)
        return {"message": "Feedback submitted", "buffer_size": len(engine.order_engine.feedback_buffer)}
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
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
