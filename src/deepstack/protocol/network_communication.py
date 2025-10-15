"""
Network communication utilities for ACPC protocol.
Ported from DeepStack Lua network_communication.lua.
"""
import socket

class ACPCNetworkCommunication:
    def __init__(self):
        self.connection = None

    def connect(self, server, port):
        self.connection = socket.create_connection((server, port))
        self._handshake()

    def _handshake(self):
        self.send_line("VERSION:2.0.0")

    def send_line(self, line):
        self.connection.sendall((line + '\r\n').encode())

    def get_line(self):
        out = b''
        while not out.endswith(b'\n'):
            chunk = self.connection.recv(1)
            if not chunk:
                raise ConnectionError("Connection closed")
            out += chunk
        return out.decode().strip()

    def close(self):
        self.connection.close()
    

def send_message(connection, message):
    """Send a message using the given connection object."""
    connection.send_line(message)
