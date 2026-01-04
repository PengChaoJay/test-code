import asyncio
import json
import websockets

async def send_binary_file(file_path):
    uri = "ws://127.0.0.1:8080/"  # WebSocket 服务器地址
    async with websockets.connect(uri) as websocket:
        # 读取二进制文件
        with open(file_path, "rb") as f:
            data = f.read()  # 读取整个文件为 bytes

        # 发送二进制数据
        await websocket.send(data)
        print(f"Sent {len(data)} bytes from {file_path}")

        # 可选：等待服务器响应
        try:
            response = await websocket.recv()
            print("Received:", response)
        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed by server.")

async def send_message():
    uri = "ws://127.0.0.1:8080/"  # WebSocket 服务器地址
    async with websockets.connect(uri) as websocket:
        data = {"answer_id":"733a1c98-a0b3-49a8-ab69-1e921fff394e","action":"stop"}
        await websocket.send(json.dumps(data))
        print(f"sent message to ws {data}")


is_stop = False 

file_path = "D:/Git/test11/111_new.data"

if is_stop:
    asyncio.run(send_message())
else:
    asyncio.run(send_binary_file(file_path))


