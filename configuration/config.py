import argparse

__all__ = ['parse_argument_bise', 'parse_arguments_federated', 'parse_arguments_nirps', 
            'parse_arguments_centralized', 'parse_arguments_fid_stats']




def parse_arguments_federated():
    parser = argparse.ArgumentParser()
    # federated setting
    parser.add_argument('--fed-aggregate-method', '-fam', type=str, default='fed-avg')
    parser.add_argument('--num-round', type=int, default=10) #10
    parser.add_argument('--num-clients', type=int, default=4) #4
    parser.add_argument('--clients-data-weight', type=float, default=None, nargs='+')
    parser.add_argument('--clip-bound', type=float, default=False)
    parser.add_argument('--noise-multiplier', type=float, default=False)
    parser.add_argument('--not-test-client', '-ntc', action='store_true', default=False)
    

    # centralized setting
    parser.add_argument('--dataset', '-d', type=str, default='synthrad', choices=['synthrad'])
    parser.add_argument('--model', '-m', type=str, default='unit', choices=['cyclegan', 'unit']) #Default= 'cyclegan'
    parser.add_argument('--source-domain', '-s', default='mri', choices=['mri'])
    parser.add_argument('--target-domain', '-t', default='ct', choices=['ct'])
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)

    parser.add_argument('--data_mode', '-dm', type=str, default='paired', choices=['paired'])
    parser.add_argument('--data-paired-weight', '-dpw', type=float, default=0.5, choices=[0.25, 0.5,1]) #Default= 0.5

    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--num-epoch', type=int, default=3) #3
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=8) #8

    # parser.add_argument('--diff-privacy', action='store_true', default=None) 
    parser.add_argument('--identity', action='store_true', default=False)
    # parser.add_argument('--reg-gan', action='store_true', default=False)
    # parser.add_argument('--fid', action='store_true', default=False)

###DATA AUGMENTATION###
    # parser.add_argument('--auxiliary-rotation', '-ar', action='store_true', default=False) ##False
    # parser.add_argument('--auxiliary-translation', '-at', action='store_true', default=False) ##False
    # parser.add_argument('--auxiliary-scaling', '-as', action='store_true', default=False)

    # parser.add_argument('--noise-level', '-nl', type=int, default=False, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--noise-type', '-nt', type=str, default=False, choices=['normal', 'slight', 'severe'])

    # input noise augmentation method
    # parser.add_argument('--severe-rotation', '-sr', type=float, default=False, choices=[15, 30, 45, 60, 90, 180])
    # parser.add_argument('--severe-translation', '-st', type=float, default=False, choices=[0.09, 0.1, 0.11])
    # parser.add_argument('--severe-scaling', '-sc', type=float, default=False, choices=[0.9, 1.1, 1.2])
    # parser.add_argument('--num-augmentation', '-na', type=str, default=False, choices=['four', 'one', 'two']) 

    parser.add_argument('--save-model', action='store_true', default=False) ##Was put false for unit
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--load-model-dir', type=str, default=None)

    parser.add_argument('--plot-distribution', action='store_true', default=False)
    parser.add_argument('--save-img', action='store_true', default=True)
    parser.add_argument('--num-img-save', type=int, default=1)
    parser.add_argument('--single-img-infer', action="store_true", default=True)

    # self-supervised augmentation
    # parser.add_argument('--angle-list', nargs='+', type=float, default=[90., 180., 270.]) #<---- added as defualt
    # parser.add_argument('--translation-list', nargs='+', type=float, default=None)
    # parser.add_argument('--scaling-list', nargs='+', type=float, default=None)

    # contraD
    # parser.add_argument('--contraD', '-cd', action='store_true', default=False)
    #parser.add_argument('--warmup', action='store_true', default=True)
    #parser.add_argument('--std-flag', action='store_true', default=False)
    #parser.add_argument('--temp', default=0.1, type=float, help='Temperature hyperparameter for contrastive losses')
    # parser.add_argument('--weight-simclr-loss', type=float)
    # parser.add_argument('--weight-supercon-loss', type=float)

    args = parser.parse_args()
    return args
    
def parse_arguments_centralized():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='synthrad', choices=['synthrad'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan'])
    parser.add_argument('--source-domain', '-s', default='mri', choices=['mri'])
    parser.add_argument('--target-domain', '-t', default='ct', choices=['ct'])
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)

    parser.add_argument('--data_mode', '-dm', type=str, default='paired', choices=['paired'])
    parser.add_argument('--data-paired-weight', '-dpw', type=float, default=0.5, choices=[0.25, 0.25, 0.25, 0.25])

    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--num-epoch', type=int, default=30)
    parser.add_argument('--debug', action='store_true', default=False)

    # parser.add_argument('--diff-privacy', action='store_true', default=False) 
    parser.add_argument('--identity', action='store_true', default=False)
    # parser.add_argument('--reg-gan', action='store_true', default=False)
    # parser.add_argument('--fid', action='store_true', default=False)

    # FedMed-ATL 
    # parser.add_argument('--atl', action='store_true', default=False, help='indicate whether the atl flag is true or not')
    # parser.add_argument('--auxiliary-rotation', '-ar', action='store_true', default=False)
    # parser.add_argument('--auxiliary-translation', '-at', action='store_true', default=False)
    # parser.add_argument('--auxiliary-scaling', '-as', action='store_true', default=False)

    # parser.add_argument('--noise-level', '-nl', type=int, default=None, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--noise-type', '-nt', type=str, default=None, choices=['normal', 'slight', 'severe'])

    # input noise augmentation method
    # parser.add_argument('--severe-rotation', '-sr', type=float, default=None, choices=[15, 30, 45, 60, 90, 180])
    # parser.add_argument('--severe-translation', '-st', type=float, default=None, choices=[0.09, 0.1, 0.11])
    # parser.add_argument('--severe-scaling', '-sc', type=float, default=None, choices=[0.9, 1.1, 1.2])

    # self-supervised augmentation
    # parser.add_argument('--angle-list', nargs='+', type=float, default=None)
    # parser.add_argument('--translation-list', nargs='+', type=float, default=None)
    # parser.add_argument('--scaling-list', nargs='+', type=float, default=None)
    # parser.add_argument('--num-augmentation', '-na', type=str, default=None, choices=['four', 'one', 'two'])

    parser.add_argument('--plot-distribution', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--load-model-dir', type=str, default=None)

    parser.add_argument('--save-img', action='store_true', default=True)
    parser.add_argument('--num-img-save', type=int, default=3)
    parser.add_argument('--single-img-infer', action='store_true', default=True)


    # contraD
    # parser.add_argument('--contraD', '-cd', action='store_true', default=False)
    # #parser.add_argument('--warmup', action='store_true', default=False)
    # #parser.add_argument('--std-flag', action='store_true', default=False)
    # parser.add_argument('--temp', default=None, type=float, help='Temperature hyperparameter for contrastive losses')
    # parser.add_argument('--weight-simclr-loss', type=float)
    # parser.add_argument('--weight-supercon-loss', type=float)

    args = parser.parse_args()
    return args


