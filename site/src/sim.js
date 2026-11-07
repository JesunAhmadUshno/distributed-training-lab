// Browser ports of the repo's stdlib Python simulations. Same arithmetic, same order:
//   level-1-learn/data_parallel.py, level-1-learn/sweeps.py,
//   level-2-build/sharding.py, level-2-build/collectives.py.
// sim.check.mjs runs at build time and asserts these reproduce the committed results files.

// ---------- Learn: 3-parameter linear model, data parallel ----------
export const learnDataset = () =>
  [[1, 0, 3], [0, 1, 0], [1, 1, 2], [2, 0, 5], [0, 2, -1], [2, 1, 4], [1, 2, 1], [3, 1, 6]].map(([a, b, y]) => ({ x: [a, b], y }));

// The dataset follows y = 2*x1 - 1*x2 + 1 exactly (build_dataset docstring); sim.check.mjs verifies every row.
export const TRUE_PARAMS = [2, -1, 1];
const predictL = (p, x) => p[0] * x[0] + p[1] * x[1] + p[2];
export function lossL(p, ex) {
  let t = 0;
  for (const e of ex) { const err = predictL(p, e.x) - e.y; t += err * err; }
  return t / ex.length;
}
export function gradL(p, ex) {
  const g = [0, 0, 0];
  for (const e of ex) {
    const err = predictL(p, e.x) - e.y;
    g[0] += 2 * err * e.x[0]; g[1] += 2 * err * e.x[1]; g[2] += 2 * err;
  }
  return g.map((v) => v / ex.length);
}
export function shardDataset(ex, n) {
  const base = Math.floor(ex.length / n), extra = ex.length % n, out = [];
  let s = 0;
  for (let r = 0; r < n; r++) { const size = base + (r < extra ? 1 : 0); out.push(ex.slice(s, s + size)); s += size; }
  return out;
}
export function allReduceAverage(gs) {
  const out = new Array(gs[0].length).fill(0);
  for (const g of gs) for (let i = 0; i < out.length; i++) out[i] += g[i];
  return out.map((v) => v / gs.length);
}
export function weightedAverage(gs, w) {
  const total = w.reduce((a, b) => a + b, 0), out = new Array(gs[0].length).fill(0);
  gs.forEach((g, k) => { for (let i = 0; i < out.length; i++) out[i] += g[i] * w[k]; });
  return out.map((v) => v / total);
}
export const maxAbsDiff = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

export function trainDataParallel({ ranks = 2, steps = 20, lr = 0.05 } = {}) {
  const ex = learnDataset(), shards = shardDataset(ex, ranks);
  let replicas = Array.from({ length: ranks }, () => [0, 0, 0]);
  const log = [];
  let maxGrad = 0, maxReplica = 0;
  for (let step = 1; step <= steps; step++) {
    const rankGrads = shards.map((s, r) => gradL(replicas[r], s));
    const rankLosses = shards.map((s, r) => lossL(replicas[r], s));
    const avg = allReduceAverage(rankGrads), full = gradL(replicas[0], ex);
    const gd = maxAbsDiff(avg, full);
    maxGrad = Math.max(maxGrad, gd);
    const combined = lossL(replicas[0], ex);
    replicas = replicas.map((p) => p.map((v, i) => v - lr * avg[i]));
    let rd = 0;
    for (let r = 1; r < ranks; r++) rd = Math.max(rd, maxAbsDiff(replicas[0], replicas[r]));
    maxReplica = Math.max(maxReplica, rd);
    log.push({ step, rankLosses, combined, gradDiff: gd, params: [...replicas[0]] });
  }
  return { log, shardSizes: shards.map((s) => s.length), finalParams: replicas[0], finalLoss: lossL(replicas[0], ex), maxGrad, maxReplica };
}

export function splitCompare(k, params = [0, 0, 0]) {
  const ex = learnDataset(), shards = [ex.slice(0, k), ex.slice(k)];
  const gs = shards.map((s) => gradL(params, s));
  const naive = allReduceAverage(gs), weighted = weightedAverage(gs, shards.map((s) => s.length)), full = gradL(params, ex);
  return { naive, weighted, full, rankGrads: gs, naiveDiff: maxAbsDiff(naive, full), weightedDiff: maxAbsDiff(weighted, full) };
}

// ---------- Build: 31-parameter MLP, sharded training ----------
export const NUM_PARAMS = 31;
export function mlpDataset() {
  const out = [];
  for (let i = 0; i < 16; i++) {
    const x = [((i % 4) - 1.5) / 1.5, (Math.floor(i / 4) - 1.5) / 1.5, (((i * 3) % 5) - 2) / 2.0, ((i * 7) % 3) - 1];
    out.push({ x, y: 2.0 * x[0] - 1.0 * x[1] + 0.5 * x[2] + 1.0 * x[3] + 0.5 });
  }
  return out;
}
export const initialParams = () => Array.from({ length: NUM_PARAMS }, (_, k) => 0.05 * (((k * 11) % 7) - 3));
function forward(p, x) {
  const h = [];
  for (let j = 0; j < 5; j++) {
    let pre = p[20 + j];
    for (let i = 0; i < 4; i++) pre += x[i] * p[i * 5 + j];
    h.push(Math.tanh(pre));
  }
  let out = p[30];
  for (let j = 0; j < 5; j++) out += h[j] * p[25 + j];
  return [out, h];
}
export function lossM(p, ex) {
  let t = 0;
  for (const e of ex) { const err = forward(p, e.x)[0] - e.y; t += err * err; }
  return t / ex.length;
}
export function gradM(p, ex) {
  const g = new Array(NUM_PARAMS).fill(0), n = ex.length;
  for (const e of ex) {
    const [pred, h] = forward(p, e.x);
    const d = 2.0 * (pred - e.y) / n;
    for (let j = 0; j < 5; j++) g[25 + j] += d * h[j];
    g[30] += d;
    for (let j = 0; j < 5; j++) {
      const dp = d * p[25 + j] * (1.0 - h[j] * h[j]);
      g[20 + j] += dp;
      for (let i = 0; i < 4; i++) g[i * 5 + j] += dp * e.x[i];
    }
  }
  return g;
}
export function partitionIndices(P, n) {
  const base = Math.floor(P / n), extra = P % n, parts = [];
  let s = 0;
  for (let r = 0; r < n; r++) { const size = base + (r < extra ? 1 : 0); parts.push(Array.from({ length: size }, (_, i) => s + i)); s += size; }
  return parts;
}
const shardVector = (v, parts) => parts.map((idx) => idx.map((i) => v[i]));
function allGather(shards, parts) {
  const out = new Array(NUM_PARAMS);
  shards.forEach((s, r) => parts[r].forEach((i, k) => { out[i] = s[k]; }));
  return out;
}

// One sharded trainer. State lives only in param and velocity shards, as in sharding.ShardedTrainer.
export class ShardedTrainer {
  constructor({ strategy = 'fsdp', ranks = 2, steps = 30, lr = 0.05, momentum = 0.5 } = {}) {
    this.config = { strategy, num_params: NUM_PARAMS, num_ranks: ranks, steps, learning_rate: lr, momentum };
    this.ex = mlpDataset();
    this.parts = partitionIndices(NUM_PARAMS, ranks);
    this.paramShards = shardVector(initialParams(), this.parts);
    this.velocityShards = this.parts.map((p) => p.map(() => 0));
    this.step = 0;
    this.losses = [];
  }
  fullParams() { return allGather(this.paramShards, this.parts); }
  runStep() {
    const full = this.fullParams();
    this.losses.push(lossM(full, this.ex));
    const gShards = shardVector(gradM(full, this.ex), this.parts);
    const { momentum: m, learning_rate: lr } = this.config;
    this.paramShards.forEach((ps, r) => ps.forEach((_, k) => {
      const v = m * this.velocityShards[r][k] - lr * gShards[r][k];
      this.velocityShards[r][k] = v;
      ps[k] += v;
    }));
    this.step += 1;
  }
  runTo(t) { while (this.step < t) this.runStep(); return this; }
  // Mirrors save_checkpoint / from_checkpoint: a JSON text round trip into a fresh trainer.
  saveCheckpoint() { return JSON.stringify({ step: this.step, param_shards: this.paramShards, velocity_shards: this.velocityShards, config: this.config }); }
  static fromCheckpoint(text) {
    const c = JSON.parse(text), cf = c.config;
    const t = new ShardedTrainer({ strategy: cf.strategy, ranks: cf.num_ranks, steps: cf.steps, lr: cf.learning_rate, momentum: cf.momentum });
    t.paramShards = c.param_shards.map((s) => [...s]);
    t.velocityShards = c.velocity_shards.map((s) => [...s]);
    t.step = c.step;
    return t;
  }
}

export function restartDemo({ strategy = 'fsdp', ckptStep = 12, ranks = 2, steps = 30 } = {}) {
  const straight = new ShardedTrainer({ strategy, ranks, steps }).runTo(steps);
  const first = new ShardedTrainer({ strategy, ranks, steps }).runTo(ckptStep);
  const text = first.saveCheckpoint();
  const resumed = ShardedTrainer.fromCheckpoint(text).runTo(steps);
  return { diff: maxAbsDiff(resumed.fullParams(), straight.fullParams()), bytes: text.length, finalLoss: lossM(straight.fullParams(), straight.ex), straight, resumedLosses: resumed.losses };
}

// ---------- Accounting formulas (sharding.py, data_parallel.py, collectives.py) ----------
export function accounting(P, N) {
  const sizes = partitionIndices(P, N).map((p) => p.length), s = Math.max(...sizes);
  const allGatherB = sizes.reduce((a, x) => a + (P - x) * 4, 0);
  const reduceScatterB = sizes.reduce((a, x) => a + x * 4, 0);
  return {
    sizes,
    allGatherB, reduceScatterB, shardedPerStepB: allGatherB + reduceScatterB,
    learnAllReduceB: P * 4 * N,
    persistent: sizes.map((x) => 2 * x),
    peak: P + 3 * s,
    baseline: 3 * P,
    stage: { ddp: 3 * P, zero1: 2 * P + s, zero2: P + 2 * s, zero3_fsdp_persistent: 3 * s, zero3_fsdp_peak: P + 3 * s },
    ringPerRank: (2 * (N - 1) / N) * P,
  };
}
