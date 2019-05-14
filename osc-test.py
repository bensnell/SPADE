from pythonosc import osc_server
from pythonosc import dispatcher


def saveTrackedLocation(unused_addr, args, x, y, z):
	print(x, y, z)

dis = dispatcher.Dispatcher()
dis.map("/rtls", saveTrackedLocation)
# server = osc_server.ThreadingOSCUDPServer(("127.0.0.1",8282), dis)
# server.serve_forever()

server = osc_server.AsyncOSCUDPServer(("127.0.0.1",8282), dis)
transport, protocol = await server.create_serve_endpoint()