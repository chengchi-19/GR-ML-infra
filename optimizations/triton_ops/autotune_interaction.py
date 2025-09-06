#!/usr/bin/env python3
import argparse, time, json, torch
from interaction_wrapper import interaction_op

def benchmark(B,F,D,block,iters=100):
    emb = torch.randn(B,F,D, device='cuda', dtype=torch.float16)
    for _ in range(10):
        _ = interaction_op(emb, BLOCK=block)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = interaction_op(emb, BLOCK=block)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0)/iters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=8)
    parser.add_argument('--F', type=int, default=16)
    parser.add_argument('--D', type=int, default=64)
    parser.add_argument('--blocks', type=str, default='32,64,128')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--out', type=str, default='autotune_result.json')
    args = parser.parse_args()
    blocks = [int(x) for x in args.blocks.split(',')]
    res = {}
    for b in blocks:
        try:
            lat = benchmark(args.B, args.F, args.D, b, iters=args.iters)
            res[b] = lat
            print('BLOCK', b, 'lat', lat)
        except Exception as e:
            res[b] = None
            print('BLOCK', b, 'error', e)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print('Wrote', args.out)
