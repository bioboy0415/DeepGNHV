from models.DeepGNHV import DeepGNHV

def build_model(config, logger):
    if config.MODEL.Model_Name == 'DeepGNHV':
        model = DeepGNHV(Graph_Type         = config.MODEL.DeepGNHV.Graph_Type
                        , GNN_Type          = config.MODEL.DeepGNHV.GNN_Type
                        , Attention_Heads   = config.MODEL.DeepGNHV.Attention_Heads
                        , Dropout_Rates     = config.MODEL.DeepGNHV.Dropout_Rates
                        , Dimensions        = config.MODEL.DeepGNHV.Dimensions
                        , num_classes       = 2
                        , num_GNN_Layers    = config.MODEL.DeepGNHV.num_GNN_Layers
                        , num_MLP_Layers    = config.MODEL.DeepGNHV.num_MLP_Layers
                        , Concat            = config.MODEL.DeepGNHV.Concat
                        , GNNBN             = config.MODEL.DeepGNHV.GNNBN
                        , GNNDP             = config.MODEL.DeepGNHV.GNNDP
                        , MLPBN             = config.MODEL.DeepGNHV.MLPBN
                        , MLPDP             = config.MODEL.DeepGNHV.MLPDP
                        , Pool_Type         = config.MODEL.DeepGNHV.Pool_Type
                        , MLPDecrease       = config.MODEL.DeepGNHV.MLPDecrease
                        )
        logger.info(f'------------------------------------------------------------------------------------')
        logger.info(f'Graph_Type:       {config.MODEL.DeepGNHV.Graph_Type}')
        logger.info(f'GNN_Type:         {config.MODEL.DeepGNHV.GNN_Type}')
        logger.info(f'Attention_Heads:  {config.MODEL.DeepGNHV.Attention_Heads}')
        logger.info(f'Dropout_Rates:    {config.MODEL.DeepGNHV.Dropout_Rates}')
        logger.info(f'Dimensions:       {config.MODEL.DeepGNHV.Dimensions}')
        logger.info(f'num_classes:      2')
        logger.info(f'num_GNN_Layers:   {config.MODEL.DeepGNHV.num_GNN_Layers}')
        logger.info(f'num_MLP_Layers:   {config.MODEL.DeepGNHV.num_MLP_Layers}')
        logger.info(f'Concat:           {config.MODEL.DeepGNHV.Concat}')
        logger.info(f'GNNBN:            {config.MODEL.DeepGNHV.GNNBN}')
        logger.info(f'GNNDP:            {config.MODEL.DeepGNHV.GNNDP}')
        logger.info(f'MLPBN:            {config.MODEL.DeepGNHV.MLPBN}')
        logger.info(f'MLPDP:            {config.MODEL.DeepGNHV.MLPDP}')
        logger.info(f'Pool_Type:        {config.MODEL.DeepGNHV.Pool_Type}')
        logger.info(f'MLPDecrease:      {config.MODEL.DeepGNHV.MLPDecrease}')
        logger.info(f'Adj.matrix thr. : {config.MODEL.DeepGNHV.Graph_Distance_Threshold}')
        logger.info(model)
        logger.info(f'------------------------------------------------------------------------------------')
    else:
        print('The model for loading is not available.')
    return model

