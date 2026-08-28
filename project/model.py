for param in clip_model.parameters():
    param.requires_grad = False
