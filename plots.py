import matplotlib.pyplot as plt

def generate_cpu_graph(data):
    """Genera y muestra un gráfico con los datos de uso de CPU recopilados."""
    plt.figure(figsize=(10, 5))
    plt.plot(data["timestamps"], data["cpu_usages_latest"], label='CPU Latest Usage', marker='o')
    plt.xlabel('Time (HH:MM:SS)')
    plt.ylabel('CPU Usage')
    plt.title('Latest CPU Usage Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_memory_graph(data):
    """Genera y muestra un gráfico con los datos de uso de memoria recopilados."""
    plt.figure(figsize=(10, 5))
    plt.plot(data["timestamps"], data["memory_usages_gb_latest"], label='Memory Usage (GB)', linestyle='--', marker='x')
    plt.xlabel('Time (HH:MM:SS)')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Latest Memory Usage Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_diskio_graph(data):
    """Genera y muestra un gráfico con los datos de I/O de disco recopilados, en Megabytes."""
    plt.figure(figsize=(10, 5))
    # Conversión de bytes a Megabytes para una mejor visualización
    disk_read_mb = [bytes / (1024**2) for bytes in data["disk_read_bytes"]]
    disk_write_mb = [bytes / (1024**2) for bytes in data["disk_write_bytes"]]
    
    plt.plot(data["timestamps"], disk_read_mb, label='Disk Read (MB)', marker='o')
    plt.plot(data["timestamps"], disk_write_mb, label='Disk Write (MB)', linestyle='--', marker='x')
    plt.xlabel('Time (HH:MM:SS)')
    plt.ylabel('Disk I/O (MB)')
    plt.title('Disk I/O Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def generate_network_graph(data):
    """Genera y muestra un gráfico con los datos de red recopilados."""
    plt.figure(figsize=(10, 5))
    plt.plot(data["timestamps"], data["network_rx_bytes"], label='Network Received Bytes', marker='o')
    plt.plot(data["timestamps"], data["network_tx_bytes"], label='Network Transmitted Bytes', linestyle='--', marker='x')
    plt.xlabel('Time (HH:MM:SS)')
    plt.ylabel('Network Bytes')
    plt.title('Network Traffic Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
