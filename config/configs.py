General_Config = {
    'general': {
        'batch_size': 2048,
        'phase_2_batch': 2048,
        'epochs': 15,
        'warm_epoch': 2,
        'rank_embedding_size': 8,
        'early_stopping_epoch': 2,
        'net_optim_lr': 1e-3,
        'kuairand_embedding_size': 16,
        'qb_video_embedding_size': 16,
        'ali_ccp_embedding_size': 16,
    },
}

ES_TFI_Config = {
    'ES_TFI': {
        'c': 0.5,
        'mu': 0.8,
        'gRDA_optim_lr': 1e-3,
        'dropout_rate': 0.2,
        'net_optim_lr': 1e-3,
        'interaction_fc_output_dim': 1,
        'mutation_threshold': 0.2,
        'mutation_probability': 0.5,
        'mutation_step_size': 10,       # τ in paper
        'adaptation_hyperparameter': 0.99,
        'adaptation_step_size': 10,     # ep in paper, controls 1/5 rule
        'population_size': 4
    },
    'ModelFunctioning': {
        'interaction_fc_output_dim': 16
    }
}
