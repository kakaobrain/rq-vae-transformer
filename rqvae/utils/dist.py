from dataclasses import dataclass
import datetime
import os
import collections
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel


def update_argument_parser(parser):
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument(
        '--local_rank', default=-1, type=int,
        help='Used for multi-process training. Can either be manually set ' +
             'or automatically set by using \'python -m torch.distributed.launch\'.')
    return parser


@dataclass
class DistEnv:
    world_size: int
    world_rank: int
    local_rank: int
    num_gpus: int
    master: bool
    device_name: str


def initialize(args, logger=None):

    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if args.world_size > 1:

        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["LOCAL_RANK"] = str(args.local_rank)

        print(f'[dist] Distributed: wait dist process group:{args.local_rank}')
        dist.init_process_group(backend=args.dist_backend, init_method='env://',
                                world_size=args.world_size,
                                timeout=datetime.timedelta(0, args.timeout))
        assert (args.world_size == dist.get_world_size())
        print(
            f"""[dist] Distributed: success device:{args.local_rank}, """,
            f"""{dist.get_rank()}/{dist.get_world_size()}"""
        )
        distenv = DistEnv(world_size=dist.get_world_size(),
                          world_rank=dist.get_rank(),
                          local_rank=args.local_rank,
                          num_gpus=1,
                          master=(dist.get_rank() == 0),
                          device_name=torch.cuda.get_device_name(),
                          )
    else:
        print('[dist] Single processed')
        distenv = DistEnv(1, 0, 0, torch.cuda.device_count(), True, torch.cuda.get_device_name())

    print(f'[dist] {distenv}')

    if logger is not None:
        logger.info(distenv)

    return distenv


def dataparallel_and_sync(distenv, model, find_unused_parameters=False):

    if dist.is_initialized():
        model = DistributedDataParallel(
            model, device_ids=[distenv.local_rank], output_device=distenv.local_rank,
            find_unused_parameters=find_unused_parameters
        )
        for _, param in model.state_dict().items():
            dist.broadcast(param, 0)

        dist.barrier()
    else:
        model = torch.nn.DataParallel(model)
    torch.cuda.synchronize()

    return model


def param_sync(param):
    dist.broadcast(param, 0)
    dist.barrier()
    torch.cuda.synchronize()


@torch.no_grad()
def all_gather_cat(distenv, tensor, dim=0):
    if distenv.world_size == 1:
        return tensor

    g_tensor = [torch.ones_like(tensor) for _ in range(distenv.world_size)]
    dist.all_gather(g_tensor, tensor)
    g_tensor = torch.cat(g_tensor, dim=dim)

    return g_tensor
