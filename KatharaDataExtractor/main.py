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
        result = subprocess.run(f"cd {constants.KATHARA_SCENARIO_PATH} && {constants.KATHARA_START_COMMAND}", shell=True, check=True, text=True)
        print(f"{constants.KATHARA_START_COMMAND} executed succesfully: {result}")
        
        result_docker_ps = subprocess.run(constants.DOCKER_PS_COMMAND, shell=True, check=True, text=True, capture_output=True)
        output = result_docker_ps.stdout.strip()
        
        if output:
            container_id = output.split()[0] 
            print(f"ID of container found: {container_id}")
            print(f"Sampling at rate: 1 sample / {constants.SAMPLING_RATE} seconds for {constants.SAMPLING_TIME} seconds. A total of {constants.SAMPLING_TIME / constants.SAMPLING_RATE} samples will be retrieved if no errors arise.")

            print(f"Fetching data, please wait {constants.SAMPLING_TIME} seconds.")
            start_time = time.time()
            while time.time() - start_time < constants.SAMPLING_TIME:
                

                # Dejamos esto asi para sacar pruebas
                
                if (time.time() - start_time) <= 32 and (time.time() - start_time) >= 30:
                    print("\n########## Entering apipecker part ##########\n")
                    print("\t aprox.", str((time.time() - start_time)))
                    try:

                        print("##### Showing the scenario #####")
                        result_linfo = subprocess.run(f"cd {constants.KATHARA_SCENARIO_PATH} && {constants.KATHARA_LINFO_COMMAND}", shell=True, check=True, text=True)
                        print(f"{constants.KATHARA_LINFO_COMMAND} executed succesfully: {result_linfo}")

                        print("##### Executing apipecker #####")
                        exec_result = subprocess.run(constants.KATHARA_EXEC_COMMAND, shell=True, check=True, text=True, capture_output=True)
                        # exec_result.stdout has the data I need
                        #output_lines = exec_result.stdout.strip().split('\n')

                        # Saving apipecker result in the csv file
                        #with open(f"./data/apipecker-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.csv", mode='w', newline='') as file:
                            #writer = csv.writer(file)
                            #for line in output_lines:
                                # Aquí asumimos que cada línea de la salida se puede dividir en columnas para el CSV
                                # Deberás ajustar el delimitador y el procesamiento según la salida real de apipecker
                                #writer.writerow(line.split())  
                            

                        #print(f"Output saved to apipecker_output.csv")
                        #print(exec_result.stdout)
                        if result.stderr:
                            print(f"Error: {result.stderr}")
                        print(f"{constants.KATHARA_EXEC_COMMAND} executed successfully: {exec_result}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error executing command in Kathara device: {e}")
                
                fetch_and_store_data(container_id, container_data)
                time.sleep(constants.SAMPLING_RATE)

            #plots.generate_cpu_graph(container_data)
            #plots.generate_memory_graph(container_data)
            #plots.generate_diskio_graph(container_data)
            #plots.generate_network_graph(container_data)
            export_data_to_csv(container_data, filename=f"./data/data-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.csv")
            result = subprocess.run(f"cd {constants.KATHARA_SCENARIO_PATH} && {constants.KATHARA_DESTROY_COMMAND}", shell=True, check=True, text=True)
            print(f"Successfully destroyed Kathara environment: {result}")
        else:
            print("No container was found.")

    except subprocess.CalledProcessError as e:
        print(f"Error in command: {e}")

if __name__ == '__main__':
    for _ in range(10):
        main()