


def save_checkpoint_with_wandb(model, optimizer, iter, loss, config, use_wandb, rank=0):
    """Save checkpoint and optionally log as WandB artifact"""
    if not master_process:
        return
    
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_iter_{iter}_{timestamp}.pt")
    
    # Save checkpoint
    if parallel_flag == 5 or parallel_flag == 6:
        raw_model = model
    else:
        raw_model = model.module
    
    checkpoint = {
        'iteration': iter,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss.detach().item() if loss else 0.0,
        'timestamp': timestamp,
    }
    
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved to: {checkpoint_file}")
    
    # Log as WandB artifact
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"model-checkpoint-iter-{iter}",
            type="model",
            description=f"Model checkpoint at iteration {iter}, loss: {loss.item()*grad_accum_steps:.4f}" if loss else f"Model checkpoint at iteration {iter}"
        )
        artifact.add_file(checkpoint_file)
        wandb.log_artifact(artifact)

