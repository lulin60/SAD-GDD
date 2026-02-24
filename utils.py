
import os
import torch
import torch.distributed as dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    torch.save(save_state, os.path.join(config.OUTPUT, f'checkpoint.pth'))
    logger.info(f"{save_path} saved !!!")


def auto_resume_helper(output_dir):
    if os.path.exists(os.path.join(output_dir, 'checkpoint.pth')):
        return os.path.join(output_dir, 'checkpoint.pth')
    
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt/ dist.get_world_size()

    return rt


def gather_tensor(tensor_):
    tensor = tensor_.clone().detach()
    aggregated_data = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=aggregated_data, tensor=tensor)

    final_data = torch.cat(aggregated_data, dim=0)
    return final_data

