"""Email monitor daemon — polls IMAP trigger folder for new emails and generates drafts.

Usage:
    python scripts/email_monitor.py [--poll-interval 30]
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.integrations.email_client import process_trigger_folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False


def main():
    parser = argparse.ArgumentParser(description="Email monitor for RAG draft generation")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between polling (default: 30)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Email monitor started (poll interval: %ds)", args.poll_interval)
    logger.info("Watching trigger folder for new emails...")

    while _running:
        try:
            results = process_trigger_folder()
            for r in results:
                logger.info(
                    "Draft generated for: %s (from: %s, lang: %s)",
                    r["subject"],
                    r["sender"],
                    r["result"]["detected_language"].upper(),
                )
            if not results:
                logger.debug("No new emails")
        except Exception:
            logger.exception("Error during polling cycle")

        # Sleep in small increments so we can respond to shutdown signals
        for _ in range(args.poll_interval):
            if not _running:
                break
            time.sleep(1)

    logger.info("Email monitor stopped")


if __name__ == "__main__":
    main()
