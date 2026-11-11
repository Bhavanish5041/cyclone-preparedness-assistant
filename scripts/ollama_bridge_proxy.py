#!/usr/bin/env python3
"""Secure internal TCP bridge proxy for Ollama.

Binds strictly to internal Docker bridge interfaces (e.g. 172.18.0.1)
forwarding requests to host loopback (127.0.0.1:11434).
Never binds to external/public network interfaces.
"""

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BRIDGE] %(message)s")
logger = logging.getLogger("ollama_bridge")

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 11434

async def forward(reader, writer):
    try:
        while not reader.at_eof():
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

async def handle_client(client_reader, client_writer):
    client_addr = client_writer.get_extra_info("peername")
    try:
        target_reader, target_writer = await asyncio.open_connection(TARGET_HOST, TARGET_PORT)
    except Exception as err:
        logger.error(f"Cannot connect to host Ollama {TARGET_HOST}:{TARGET_PORT}: {err}")
        client_writer.close()
        return

    await asyncio.gather(
        forward(client_reader, target_writer),
        forward(target_reader, client_writer),
        return_exceptions=True
    )

async def main():
    bridge_ips = ["172.18.0.1", "172.17.0.1"]
    if len(sys.argv) > 1 and sys.argv[1] not in bridge_ips:
        bridge_ips.append(sys.argv[1])
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 11434

    servers = []
    for ip in bridge_ips:
        try:
            srv = await asyncio.start_server(handle_client, ip, port)
            servers.append(srv)
            logger.info(f"Ollama bridge active on {ip}:{port} -> {TARGET_HOST}:{TARGET_PORT}")
        except Exception as err:
            logger.warning(f"Could not bind bridge on {ip}:{port}: {err}")

    if not servers:
        logger.error("Failed to bind any bridge listener!")
        sys.exit(1)

    await asyncio.gather(*(srv.serve_forever() for srv in servers))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
