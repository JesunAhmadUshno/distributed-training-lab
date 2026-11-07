#!/usr/bin/env bash
# Launch one cluster script on both Sparks. Node 0 is this machine (rendezvous
# host), node 1 is reached over SSH. The script folder is copied to node 1 on
# every launch, so both nodes always run identical code.
#   ./launch.sh 2 e1_correct.py --mode ddp --out ../runs/x.jsonl     (2 nodes)
#   ./launch.sh 1 e3_scale.py --mode single ...                      (node 0 only)
# Static rendezvous (the c10d elastic rendezvous hung on this pair, 2026-11-07).
# Extra torchrun flags go in TR_EXTRA (e.g. "--max-restarts=2").
set -euo pipefail
NODES=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
PEER=${PEER:-spark-31c4}
MASTER=${MASTER:-192.168.99.170}
REMOTE=${REMOTE:-p19-run/cluster}
export P19_PG_TIMEOUT=${P19_PG_TIMEOUT:-600} TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enP7s7} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enP7s7} NCCL_IB_DISABLE=1
TR="python3 -m torch.distributed.run --nproc_per_node=1 ${TR_EXTRA:-}"
if [ "$NODES" = 1 ]; then
  cd "$HERE" && exec $TR --nnodes=1 --master_port=29511 "$@"
fi
RDZV="--nnodes=2 --master_addr=$MASTER --master_port=${MASTER_PORT:-29500}"
ssh "$PEER" "mkdir -p $REMOTE"
scp -q "$HERE"/*.py "$PEER:$REMOTE/"
ENVS="P19_PG_TIMEOUT=${P19_PG_TIMEOUT:-600} TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME NCCL_IB_DISABLE=1"
# Node 1 writes nothing that matters: results are emitted by rank 0 only. Paths are mirrored so --out/--ckpt resolve.
ssh "$PEER" "cd $REMOTE && mkdir -p ../runs ../ckpt && env $ENVS $TR $RDZV --node_rank=1 $*" > "$HERE/../runs/node1.log" 2>&1 &
PEER_PID=$!
cd "$HERE"
set +e
$TR $RDZV --node_rank=0 "$@"; RC0=$?
wait $PEER_PID; RC1=$?
echo "node0 rc=$RC0 node1 rc=$RC1"
[ $RC0 = 0 ] && [ $RC1 = 0 ]
