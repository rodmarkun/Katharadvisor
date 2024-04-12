import requests
import constants
import subprocess
import time
import plots
import csv
from datetime import datetime

def fetch_and_store_data(container_id, container_data):
    url_summary = f"http://{constants.CADVISOR_DEPLOYMENT}/api/v2.0/summary/{container_id}?type=docker"
    response_summary = requests.get(url_summary)
    if response_summary.status_code == 200:
        summary_data = response_summary.json()
        container_key = next(iter(summary_data))
        summary_data = summary_data[container_key]

        # Actualiza aquí con los campos correctos según tu estructura de datos
        cpu_usage_latest = summary_data['latest_usage']['cpu']
        memory_usage_latest = summary_data['latest_usage']['memory']

        container_data["timestamps"].append(datetime.now().strftime('%H:%M:%S'))
        container_data["cpu_usages_latest"].append(cpu_usage_latest)
        container_data["memory_usages_gb_latest"].append(memory_usage_latest / (1024**3))
    else:
        print(f"Error fetching summary data: {response_summary.status_code}")

    url_stats = f"http://{constants.CADVISOR_DEPLOYMENT}/api/v2.0/stats/{container_id}?type=docker&count=1"
    response_stats = requests.get(url_stats)
    if response_stats.status_code == 200:
        stats_data = response_stats.json()
        stats_data = stats_data[container_key][0]  # Asumiendo que 'container_key' sigue siendo válido y que los datos están en el primer elemento

        # Maneja correctamente los datos de 'diskio' y red
        if 'diskio' in stats_data and stats_data['diskio']:
            diskio_data = stats_data['diskio']['io_service_bytes'][0]['stats']
            container_data["disk_read_bytes"].append(diskio_data.get('read', 0))
            container_data["disk_write_bytes"].append(diskio_data.get('write', 0))
        else:
            container_data["disk_read_bytes"].append(0)
            container_data["disk_write_bytes"].append(0)

        if 'network' in stats_data and stats_data['network']['interfaces']:
            network_data = stats_data['network']['interfaces'][0]
            container_data["network_rx_bytes"].append(network_data.get('rx_bytes', 0))
            container_data["network_tx_bytes"].append(network_data.get('tx_bytes', 0))
        else:
            container_data["network_rx_bytes"].append(0)
            container_data["network_tx_bytes"].append(0)
    else:
        print(f"Error fetching stats data: {response_stats.status_code}")


def export_data_to_csv(container_data, filename="container_metrics.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escribe los encabezados
        headers = ["timestamp", "cpu_usage_latest", "memory_usage_gb_latest", "disk_read_bytes", "disk_write_bytes", "network_rx_bytes", "network_tx_bytes"]
        writer.writerow(headers)
        
        # Escribe los datos
        for i in range(len(container_data["timestamps"])):
            row = [
                container_data["timestamps"][i],
                container_data["cpu_usages_latest"][i],
                container_data["memory_usages_gb_latest"][i],
                container_data["disk_read_bytes"][i],
                container_data["disk_write_bytes"][i],
                container_data["network_rx_bytes"][i],
                container_data["network_tx_bytes"][i]
            ]
            writer.writerow(row)



def main():
    container_data = {"timestamps": [],
            "cpu_usages_latest": [],
            "memory_usages_gb_latest": [],
            "disk_read_bytes": [],
            "disk_write_bytes": [],
            "network_rx_bytes": [],
            "network_tx_bytes": []}
    try:
        """
        builder_command = f"sudo docker buildx create --name mybuilder --use"
        result_builder = subprocess.run(builder_command, shell=True, check=True, text=True)
        print(f"Docker builder: {result_builder}")

        init_builder_command = f"sudo docker buildx inspect --bootstrap"
        result_init_builder = subprocess.run(init_builder_command, shell=True, check=True, text=True)
        print(f"Docker builder: {result_init_builder}")
        """
                
        build_command = f"sudo docker build -t my_docker_image {constants.DOCKERFILE_PATH}"
        result_build = subprocess.run(build_command, shell=True, check=True, text=True)
        print(f"Docker image built successfully: {result_build}")

        # Ejecutar un contenedor Docker a partir de la imagen construida
        run_command = f"sudo docker run -d my_docker_image"
        result_run = subprocess.run(run_command, shell=True, check=True, text=True, capture_output=True)
        container_id = result_run.stdout.strip()
        print(f"Container launched successfully with ID: {container_id}")
        
        result_docker_ps = subprocess.run(constants.DOCKER_PS_COMMAND, shell=True, check=True, text=True, capture_output=True)
        output = result_docker_ps.stdout.strip()
        
        if output:
            container_id = output.split()[0] 
            print(f"ID of container found: {container_id}")
            print(f"Sampling at rate: 1 sample / {constants.SAMPLING_RATE} seconds for {constants.SAMPLING_TIME} seconds. A total of {constants.SAMPLING_TIME / constants.SAMPLING_RATE} samples will be retrieved if no errors arise.")

            print(f"Fetching data, please wait {constants.SAMPLING_TIME} seconds.")
            start_time = time.time()
            while time.time() - start_time < constants.SAMPLING_TIME:
                fetch_and_store_data(container_id, container_data)
                time.sleep(constants.SAMPLING_RATE)

            #plots.generate_cpu_graph(container_data)
            #plots.generate_memory_graph(container_data)
            #plots.generate_diskio_graph(container_data)
            #plots.generate_network_graph(container_data)
            export_data_to_csv(container_data, filename=f"data/data-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.csv")

            cleanup_command = f"sudo docker stop {container_id} && docker rm {container_id}"
            result_cleanup = subprocess.run(cleanup_command, shell=True, check=True, text=True)
            print(f"Container stopped and removed successfully: {result_cleanup}")
        else:
            print("No container was found.")

    except subprocess.CalledProcessError as e:
        print(f"Error in command: {e}")

if __name__ == '__main__':
    for _ in range(1):
        main()