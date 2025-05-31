import torch

def get_checkpoint(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path)
    except Exception as e:
        print(e)
        return None
    return ckpt

def save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'scheduler': scheduler.state_dict(),
    }
    try:
        ckpt = torch.save(checkpoint, ckpt_path)
    except Exception as e:
        print(e)
