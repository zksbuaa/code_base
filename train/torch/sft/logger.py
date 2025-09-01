import logging
from typing import Optional
import torch.distributed as dist

class DistributedLogger:
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self.logger.setLevel(logging.DEBUG if self.is_main_process else logging.WARNING)
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            f'%(asctime)s - [RANK {self.rank}/{self.world_size}] - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if log_file and self.is_main_process:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg: str, force_all_ranks: bool = False):
        if self.is_main_process or force_all_ranks:
            self.logger.info(msg)
    
    def warning(self, msg: str, force_all_ranks: bool = False):
        if self.is_main_process or force_all_ranks:
            self.logger.warning(msg)
    
    def error(self, msg: str, force_all_ranks: bool = True):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        if self.is_main_process:
            self.logger.debug(msg)