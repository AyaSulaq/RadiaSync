import argparse

__all__ = ['parse_argument_bise', 'parse_arguments_federated', 'parse_arguments_nirps', 
            'parse_arguments_centralized', 'parse_arguments_fid_stats']




def parse_arguments_federated():
    parser = argparse.ArgumentParser()
    # federated setting
    parser.add_argument('--fed-aggregate-method', '-fam', type=str, default='fed-avg')
    parser.add_argument('--num-round', type=int, default=20) #10
    parser.add_argument('--num-clients', type=int, default=4) #4
    parser.add_argument('--clients-data-weight', type=float, default=None, nargs='+')
    parser.add_argument('--clip-bound', type=float, default=False)
    parser.add_argument('--noise-multiplier', type=float, default=False)
    parser.add_argument('--not-test-client', '-ntc', action='store_true', default=False)
    

    # centralized setting
    parser.add_argument('--dataset', '-d', type=str, default='synthrad', choices=['synthrad'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan', 'unit']) #Default= 'cyclegan'
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

    parser.add_argument('--identity', action='store_true', default=False)

###DATA AUGMENTATION###
    parser.add_argument('--noise-type', '-nt', type=str, default=False, choices=['normal', 'slight', 'severe'])

    parser.add_argument('--save-model', action='store_true', default=True) ##Was put false for unit
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--load-model-dir', type=str, default=None)

    parser.add_argument('--plot-distribution', action='store_true', default=False)
    parser.add_argument('--save-img', action='store_true', default=True)
    parser.add_argument('--num-img-save', type=int, default=3)
    parser.add_argument('--single-img-infer', action="store_true", default=True)

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

    parser.add_argument('--identity', action='store_true', default=False)
    parser.add_argument('--noise-type', '-nt', type=str, default='normal', choices=['normal', 'slight', 'severe'])

    parser.add_argument('--plot-distribution', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--load-model-dir', type=str, default=None)

    parser.add_argument('--save-img', action='store_true', default=True)
    parser.add_argument('--num-img-save', type=int, default=3)
    parser.add_argument('--single-img-infer', action='store_true', default=True)

    args = parser.parse_args()
    return args


