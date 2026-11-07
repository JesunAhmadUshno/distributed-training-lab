// Known-answer self-check, run by "npm run build" before vite: node src/sim.check.mjs
// Every browser computation is replayed here and compared with the committed results files.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { TRUE_PARAMS, learnDataset, trainDataParallel, splitCompare, ShardedTrainer, restartDemo, accounting, maxAbsDiff, partitionIndices } from './sim.js';

const read = (p) => JSON.parse(fs.readFileSync(new URL(`../../${p}`, import.meta.url), 'utf8'));
const learn = read('level-1-learn/results_learn.json');
const sweep = read('level-1-learn/results_sweep.json');
const build = read('level-2-build/results_build.json');
const scaling = read('level-2-build/results_scaling.json');
const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
const closeVec = (a, b, tol, what) => assert.ok(maxAbsDiff(a, b) <= tol, `${what}: max diff ${maxAbsDiff(a, b)}`);

for (const e of learnDataset()) assert.equal(TRUE_PARAMS[0] * e.x[0] + TRUE_PARAMS[1] * e.x[1] + TRUE_PARAMS[2], e.y);

// 1. Learn trainer at the recorded defaults: every logged step within 1e-12.
const run = trainDataParallel({ ranks: 2, lr: 0.05, steps: 20 });
learn.training.log.forEach((e, i) => {
  close(run.log[i].combined, e.combined_loss_before_update, 1e-12, `learn loss step ${e.step}`);
  closeVec(run.log[i].rankLosses, e.rank_losses, 1e-12, `learn rank losses step ${e.step}`);
  closeVec(run.log[i].params, e.params_after_update, 1e-12, `learn params step ${e.step}`);
  close(run.log[i].gradDiff, e.grad_max_abs_diff, 1e-12, `learn grad diff step ${e.step}`);
});
const ls = learn.training.summary;
close(run.finalLoss, ls.final_loss, 1e-12, 'learn final loss');
close(run.maxGrad, ls.max_grad_diff_vs_single_process, 1e-12, 'learn max grad diff');
assert.equal(run.maxReplica, ls.max_replica_diff);

// 2. Rank sweep: same trainer at 1..8 ranks.
for (const r of sweep.rank_sweep) {
  const t = trainDataParallel({ ranks: r.num_ranks });
  assert.deepEqual(t.shardSizes, r.shard_sizes);
  close(t.maxGrad, r.max_grad_diff_vs_single_process, 1e-12, `sweep grad diff N=${r.num_ranks}`);
  close(t.finalLoss, r.final_loss, 1e-12, `sweep final loss N=${r.num_ranks}`);
}

// 3. Shard imbalance: the recorded 6|2 experiment and every split in the sweep.
const imb = learn.imbalance_experiment, six = splitCompare(6);
close(six.naiveDiff, imb.naive_max_abs_diff_vs_full, 1e-12, 'imbalance 6|2 naive');
close(six.weightedDiff, imb.weighted_max_abs_diff_vs_full, 1e-12, 'imbalance 6|2 weighted');
for (const r of sweep.imbalance_sweep) {
  const c = splitCompare(r.shard_sizes[0]);
  close(c.naiveDiff, r.naive_max_abs_diff_vs_full, 1e-12, `split ${r.shard_sizes} naive`);
  close(c.weightedDiff, r.weighted_max_abs_diff_vs_full, 1e-12, `split ${r.shard_sizes} weighted`);
}

// 4. Sharded MLP, both strategies, 2 ranks, 30 steps. Math.tanh may differ from CPython by an ulp, hence 1e-12.
for (const strategy of ['fsdp', 'zero']) {
  const rec = build.strategies[strategy];
  const t = new ShardedTrainer({ strategy, ranks: 2, steps: 30 }).runTo(30);
  closeVec(t.fullParams(), rec.final_params, 1e-12, `${strategy} final params`);
  rec.log.forEach((e, i) => close(t.losses[i], e.loss_before_update, 1e-12, `${strategy} loss step ${e.step}`));
  assert.deepEqual(t.parts.map((p) => p.length), rec.shard_sizes);
  // 5. Checkpoint at the recorded step, rebuilt from JSON, matches the uninterrupted run.
  const r = restartDemo({ strategy, ckptStep: rec.checkpoint_step });
  assert.ok(r.diff <= rec.restart_max_diff_vs_uninterrupted + 1e-12, `${strategy} restart diff ${r.diff}`);
}

// 6. Accounting formulas against the recorded 2-rank values.
const a = accounting(31, 2), fc = build.strategies.fsdp.communication, fm = build.strategies.fsdp.memory;
assert.equal(a.allGatherB, fc.all_gather_bytes_per_step_all_ranks);
assert.equal(a.reduceScatterB, fc.reduce_scatter_bytes_per_step_all_ranks);
assert.equal(a.shardedPerStepB, fc.bytes_per_step_all_ranks);
assert.equal(a.shardedPerStepB * 30, fc.total_bytes_all_ranks);
assert.deepEqual(a.persistent, fm.persistent_elements_per_rank);
assert.equal(a.peak, fm.peak_elements_max_rank);
assert.equal(a.baseline, fm.baseline_full_elements_per_rank);
assert.equal(accounting(3, 2).learnAllReduceB * 20, ls.communication.total_bytes_all_ranks);
for (const m of scaling.stage_memory) {
  const s = accounting(m.num_params, m.num_ranks).stage;
  for (const k of Object.keys(s)) assert.equal(s[k], m[k], `stage ${k} N=${m.num_ranks}`);
}
for (const r of scaling.ring_all_reduce) close(accounting(31, r.num_ranks).ringPerRank, r.ideal_elements_per_rank, 1e-12, `ring ideal N=${r.num_ranks}`);
assert.equal(partitionIndices(31, 2).flat().length, 31);

console.log('sim ok', { learnFinalLoss: run.finalLoss, learnMaxGradDiff: run.maxGrad, splits: sweep.imbalance_sweep.length, ranks: sweep.rank_sweep.length });
