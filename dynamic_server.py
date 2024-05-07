import asyncio
import websockets

# Store the values received from the clients
values = {}
indexes = {}
weights = {}
correction = {}
first = True


async def handler(websocket, path):
    global values  # Use the global 'values' dictionary
    global weights
    global indexes
    global first

    while True:
        try:
            # Receive the value from the client
            value = await websocket.recv()
            print(f"Received value from client {id(websocket)}, value: {value}")

            # Store the value received from the client
            if first:
                values[websocket] = value  # Convert the value to float for comparison
            else:
                # index_weight_correction
                split_value = value.split("_")
                indexes[websocket] = int(split_value[0])
                weights[websocket] = int(split_value[1])
                correction[websocket] = float(split_value[2])

            # If we have received values from all 4 clients
            if len(values) == 2 and first:
                print("Sending Responses")

                # Standby -- wait for user input
                string = "no"
                while string != "yes":
                    string = input("Want to start a game? (yes/no): ")
                    if string == "yes":
                        first = False
                        break

                        # Send 'yes' to the client with the maximum value and 'no' to the others
                for client in values:
                    await client.send('start')

                    # Clear the 'values' dictionary for the next round
                print('Responses Sent')
                values = {}
            elif len(indexes) == 2:
                # Find the client with the maximum value
                print('w', weights)
                print()
                print('i', indexes)
                max_client = max(weights, key=weights.get)
                max_cor = 0
                for client in weights:
                    if client is max_client:
                        max_cor = correction[client]
                        await client.send(f"{indexes[client]}_main_0")
                    else:
                        await client.send(f"{indexes[client]}_sec_{max_cor}")
                weights = {}
                indexes = {}

            print("ii", len(indexes))
            print("ww", len(weights))
            print("vv", len(values))

        except Exception:
            pass


start_server = websockets.serve(handler, port=6969)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
