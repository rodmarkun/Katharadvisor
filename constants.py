# CONFIGURE:
KATHARA_SCENARIO_PATH = "/home/pablospilab/Pascal/spilab-pascal-769625c67efb/monitest"
KATHARA_DEVICE_NAME = "device1" 
KATHARA_IMAGE = "pascal/android"
CADVISOR_DEPLOYMENT = "localhost:8080"
SAMPLING_RATE = 1 # In seconds, minimum: 1
SAMPLING_TIME = 15 # In seconds, minimum: 1

# NO NEED TO CHANGE:
KATHARA_START_COMMAND = "sudo kathara lstart --privileged"
DOCKER_PS_COMMAND = f"sudo docker ps | grep {KATHARA_DEVICE_NAME} | grep {KATHARA_IMAGE}" if KATHARA_IMAGE is not None else f"sudo docker ps | grep {KATHARA_DEVICE_NAME}"
KATHARA_DESTROY_COMMAND = "sudo kathara wipe -a -f"