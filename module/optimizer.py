from torch import optim as optim

def build_optimizer(config, model, logger):
    Optimizer      = None
    parameters     = model.parameters()
    Optimizer_Name = config.Train.Optimizer.Name.lower()
    
    logger.info(f'------------------------------------------------------------------------------------')
    logger.info(f'Optimizer Type:           {config.Train.Optimizer.Name}')
    logger.info(f'Optimizer LR:             {config.Train.Lr}')
    logger.info(f'Optimizer Min LR:         {config.Train.Min_Lr}')
    logger.info(f'Optimizer weight decay:   {config.Train.Optimizer.Weight_Decay}')
    logger.info(f'Optimizer momentum:       {config.Train.Optimizer.Momentum}')
    logger.info(f'------------------------------------------------------------------------------------')
    
    if Optimizer_Name == 'sgd':
        Optimizer = optim.SGD(parameters
                            , momentum     = config.Train.Optimizer.Momentum
                            , lr           = config.Train.Lr
                            , weight_decay = config.Train.Optimizer.Weight_Decay)
    elif Optimizer_Name == 'adamw':
        Optimizer = optim.AdamW(parameters
                                , lr           = config.Train.Lr
                                , eps          = config.Train.Optimizer.Eps
                                , betas        = config.Train.Optimizer.Betas
                                , weight_decay = config.Train.Optimizer.Weight_Decay
                                )
    elif Optimizer_Name == 'adam':
        Optimizer = optim.Adam(parameters
                               , lr           = config.Train.Lr
                               , weight_decay = config.Train.Optimizer.Weight_Decay)

    return Optimizer
