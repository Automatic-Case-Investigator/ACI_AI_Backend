# ACI AI Backend

This is the local AI backend designed for running models that powers automatic case investigations.

Major functionalities include
- **Task Generation:** Generates investigation tasks based on the provided case information on the SOAR platform
- **Activity Generation:** Generates investigation subtasks based on the provided case information on the SOAR platform
- **Query Generation:** Gathers evidences by performing SIEM queries relevant to the subtask if possible, or otherwise gather evidences from the case information itself

## Supported Platforms

### SOAR
- [The Hive](https://strangebee.com/)

### SIEM
- [Wazuh](https://wazuh.com/)

## Installation

### Using Docker

1. Copy the sample environment file and customize it to your setup:
   ```bash
   cp sample.env .env
   ```


2. Build and run the docker compose project:
  
    **Linux / Mac:**
    ```bash
    sudo docker compose -f docker-compose.yml build
    sudo docker compose -f docker-compose.yml up
    ```
    
    **Windows:**
    ```bash
    sudo docker compose -f docker-compose-windows.yml build
    sudo docker compose -f docker-compose-windows.yml up
    ```
