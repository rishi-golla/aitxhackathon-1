"""
OSHA Vision - Factory Safety Copilot

Main entry point for the safety monitoring system.
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from src.utils.logging_config import setup_logging
from src.utils.config import ConfigManager, get_config, create_default_config
from src.pipeline.orchestrator import create_pipeline, SafetyPipelineOrchestrator

log = structlog.get_logger()


async def main():
    """Main entry point."""
    # Setup logging
    setup_logging(level="INFO", log_file="logs/app.log")

    log.info("osha_vision_starting", version="1.0.0")

    # Load configuration
    config_path = "config/app.yaml"
    if not Path(config_path).exists():
        log.info("creating_default_config")
        create_default_config(config_path)

    config = get_config()
    log.info("config_loaded", cameras=len(config.cameras))

    # Create pipeline
    pipeline = await create_pipeline(config)

    # Setup shutdown handler
    shutdown_event = asyncio.Event()

    def signal_handler():
        log.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Start pipeline
    try:
        await pipeline.start()
        log.info("pipeline_running", status="active")

        # Print status periodically
        while not shutdown_event.is_set():
            await asyncio.sleep(5)
            status = pipeline.get_status()
            log.info(
                "pipeline_status",
                fps=status["stats"]["fps"],
                frames=status["stats"]["frames_processed"],
                violations=status["stats"]["violations_detected"]
            )

    except Exception as e:
        log.error("pipeline_error", error=str(e))
    finally:
        await pipeline.stop()
        log.info("osha_vision_stopped")


def run_cli():
    """CLI entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         OSHA Vision - Factory Safety Copilot              ║
    ║                                                           ║
    ║  Real-time PPE detection and safety compliance monitoring ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == "__main__":
    run_cli()
