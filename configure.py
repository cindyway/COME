def get_default_config(data_name):
    if data_name == 'dro':
        return dict(
            dims=[84, 256, 128],
            pretrain_epochs=500,
            epochs=110,
            pre_lr=0.001,
            lr=0.001,
            batch_size=256,
            weight_map=0.1,
            weight_coef=1e-6,
            weight_ssim=0.01,
            weight_con=0.01,
            tau=0.8,

        )

    elif data_name == 'MERFISH':
        return dict(
            dims=[232, 512,256],
            pretrain_epochs=500,
            epochs=200,
            pre_lr=0.01,
            lr=0.001,
            batch_size=128,
            weight_map=1,
            weight_coef=0.001,
            weight_ssim=1,
            weight_con=1,
            tau=0.8,

        )
    elif data_name == 'smFISH':
        return dict(
            dims=[22, 256,128],
            pretrain_epochs=500,
            epochs=100,
            pre_lr=0.001,
            lr=0.001,
            batch_size=256,
            weight_map=10,
            weight_coef=0.01,
            weight_ssim=1,
            weight_con=1,
            tau=0.8,

        )
    elif data_name == 'STARmap':
        return dict(
            dims=[996, 512,256],
            pretrain_epochs=500,
            epochs=150,
            pre_lr=0.001,
            lr=0.001,
            batch_size=256,
            weight_map=1,
            weight_coef=0.001,
            weight_ssim=1,
            weight_con=1,
            tau=0.8,

        )

    elif data_name == 'PDAC':
        return dict(
            dims=[3000, 2048, 256 ],
            pretrain_epochs=500,
            epochs=1000,
            pre_lr=0.001,
            lr=0.001,
            batch_size=128,
            weight_map=1,
            weight_coef=0.001,
            weight_ssim=1,
            weight_con=10,
            tau=0.8,
        )

    else:
        raise Exception('Undefined data_name')
