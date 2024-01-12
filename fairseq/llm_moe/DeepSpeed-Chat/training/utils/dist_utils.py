import torch.distributed as dist


def _is_initialized():
    if dist.is_available() and dist.is_initialized():
        return True
    else:
        return False


def is_main_process():
    if _is_initialized():
        if dist.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def wait_for_everyone():
    if _is_initialized():
        dist.barrier()
    else:
        return