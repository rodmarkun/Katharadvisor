# CONFIGURE:
KATHARA_SCENARIO_PATH = "/home/eduardo/pruebas-pascal/cu_con_internet/testmovil3"
KATHARA_DEVICE_NAME = "movil2" 
KATHARA_IMAGE = "pascal/simple"
CADVISOR_DEPLOYMENT = "localhost:8080"
SAMPLING_RATE = 1 # In seconds, minimum: 1
SAMPLING_TIME = 60 # In seconds, minimum: 1

# NO NEED TO CHANGE:
KATHARA_START_COMMAND = "sudo kathara lstart --privileged"
KATHARA_LINFO_COMMAND = "sudo kathara linfo"
KATHARA_CONNECT_S1_COMMAND = "sudo kathara connect s1"
# APIPECKER_COMMAND = "npx apipecker 20 2 580 http://8.0.1.2 -v"
KATHARA_EXEC_COMMAND = "sudo kathara exec -d /home/eduardo/pruebas-pascal/cu_con_internet/testmovil3 movil1 -- npx apipecker 20 2 580 http://8.0.0.2:8080 -v"
DOCKER_PS_COMMAND = f"sudo docker ps | grep {KATHARA_DEVICE_NAME} | grep {KATHARA_IMAGE}" if KATHARA_IMAGE is not None else f"sudo docker ps | grep {KATHARA_DEVICE_NAME}"
KATHARA_DESTROY_COMMAND = "sudo kathara wipe -a -f"