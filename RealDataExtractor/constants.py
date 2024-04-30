# CONFIGURE:
KATHARA_SCENARIO_PATH = "/home/eduardo/pruebas-pascal/cu_con_internet/teststress2"
KATHARA_DEVICE_NAME = "movil2" 
DOCKER_IMAGE = "my_docker_image"
DOCKERFILE_PATH = "./images"
CADVISOR_DEPLOYMENT = "localhost:8080"
SAMPLING_RATE = 1 # In seconds, minimum: 1
SAMPLING_TIME = 90 # In seconds, minimum: 1

# NO NEED TO CHANGE:
KATHARA_START_COMMAND = "sudo kathara lstart --privileged"
DOCKER_PS_COMMAND = f"sudo docker ps | grep {DOCKER_IMAGE}"
DOCKER_STOP_COMMAND = f"sudo docker stop {DOCKER_IMAGE}"
DOCKER_RM_COMMAND = f"sudo docker rm {DOCKER_IMAGE}"
KATHARA_DESTROY_COMMAND = "sudo kathara wipe -a -f"
APIPECKER_EXEC_COMMAND = "apipecker 3000 3 580 http://localhost:8081/"