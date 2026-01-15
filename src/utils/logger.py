import logging
import os
import sys
from datetime import datetime


def setup_logging(log_dir: str = "logs", log_filename: str = "app_rag.log"):
    """
    Configura o sistema de logging para escrever em arquivo E no terminal.

    Args:
        log_dir (str): Diretório onde os logs serão salvos.
        log_filename (str): Nome do arquivo de log.
    """
    # 1. Cria a pasta de logs se não existir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Caminho completo do arquivo (ex: logs/app_rag.log)
    # Dica: Você pode adicionar data no nome se quiser um arquivo novo por execução:
    # file_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}_{log_filename}")
    file_path = os.path.join(log_dir, log_filename)

    # 2. Define o formato da mensagem
    # %(asctime)s - Data/Hora
    # %(name)s    - Nome do módulo (ex: src.components.llm)
    # %(levelname)s - Nível (INFO, ERROR, DEBUG)
    # %(message)s - A mensagem em si
    log_format = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 3. Cria os Handlers (Manipuladores)

    # Handler do Arquivo (FileHandler)
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Handler do Terminal (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Pode mudar para DEBUG para ver mais detalhes no terminal
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # 4. Aplica a configuração global (Root Logger)
    # force=True garante que estamos sobrescrevendo configs anteriores
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True
    )

    # Cria um logger local só para avisar que deu certo
    logger = logging.getLogger(__name__)
    logger.info(f"✅ Sistema de Logs inicializado.")
    logger.info(f"📂 Logs sendo salvos em: {os.path.abspath(file_path)}")