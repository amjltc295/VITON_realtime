import socket
import pickle

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
print(host)
serversocket.bind(('', 8089))
serversocket.listen(2)  # become a server socket, maximum 5 connections

while True:
    connection, address = serversocket.accept()
    buf = connection.recv(100000000)
    image = pickle.loads(buf)
    viton_data = {'frame': image}
    viton_data_pickle = pickle.dumps(buf)
    connection.send(viton_data_pickle)
