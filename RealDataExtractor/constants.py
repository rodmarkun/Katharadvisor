# CONFIGURE:
KATHARA_SCENARIO_PATH = "/home/pablospilab/Pascal/spilab-pascal-769625c67efb/monitest"
KATHARA_DEVICE_NAME = "device1" 
DOCKER_IMAGE = "my_docker_image"
DOCKERFILE_PATH = "./images"
CADVISOR_DEPLOYMENT = "localhost:8080"
SAMPLING_RATE = 1 # In seconds, minimum: 1
SAMPLING_TIME = 320 # In seconds, minimum: 1

# NO NEED TO CHANGE:
KATHARA_START_COMMAND = "sudo kathara lstart --privileged"
DOCKER_PS_COMMAND = f"sudo docker ps | grep {DOCKER_IMAGE}"
DOCKER_STOP_COMMAND = f"sudo docker stop {DOCKER_IMAGE}"
DOCKER_RM_COMMAND = f"sudo docker rm {DOCKER_IMAGE}"
KATHARA_DESTROY_COMMAND = "sudo kathara wipe -a -f"