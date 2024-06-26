from configuration.config import parse_arguments_federated
from arch_federated.fed_cyclegan import FedCycleGAN
# from arch_federated.fed_munit import FedMunit
from arch_federated.fed_unit import FedUnit



def federated_training():
    args = parse_arguments_federated()

    for key_arg in ['dataset', 'model', 'source_domain', 'target_domain']:
        if not vars(args)[key_arg]:
            raise ValueError('Parameter {} Must Be Refered!'.format(key_arg))

    if args.model == 'cyclegan':
         work = FedCycleGAN(args=args)
    # if args.model == 'unit':
    #    work = FedUnit(args=args)
    # else:
    #     raise ValueError('Model Is Invalid!')   
    
    work.run_work_flow()




if __name__ == '__main__':
    federated_training()