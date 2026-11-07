import { useMemo, useState, useEffect, useId } from 'react';
import learn from '../../level-1-learn/results_learn.json';
import sweep from '../../level-1-learn/results_sweep.json';
import build from '../../level-2-build/results_build.json';
import scaling from '../../level-2-build/results_scaling.json';
import deep from '../../level-3-deep/results_deep_prep.json';
import { TRUE_PARAMS, trainDataParallel, splitCompare, ShardedTrainer, restartDemo, accounting, partitionIndices, lossM, NUM_PARAMS } from './sim.js';

// Source files read at build time: the Kaggle scripts (for their purpose lines) and the test files (for the test count).
const kaggle = import.meta.glob('../../level-3-deep/kaggle/*.py', { query: '?raw', import: 'default', eager: true });
const testFiles = import.meta.glob('../../tests/*.py', { query: '?raw', import: 'default', eager: true });
const testCount = Object.values(testFiles).reduce((n, src) => n + (src.match(/^\s+def test_/gm) || []).length, 0);

const REPO = 'https://github.com/JesunAhmadUshno/distributed-training-lab';
const blob = (p) => `${REPO}/blob/main/${p}`;
const e2 = (x) => (x === 0 ? '0' : x.toExponential(2));
const f6 = (x) => x.toFixed(6);
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const LS = learn.training.summary;
const FS = build.strategies.fsdp, ZS = build.strategies.zero;
const RANK_COLORS = ['#2457d6', '#e0793a', '#1f8f5f', '#a04ad6', '#c53a3a', '#a46b00', '#2a9bb5', '#6f6a61'];

const SECTIONS = [['top', 'Overview', '◎'], ['learn', 'Data parallel', '⇉'], ['imbalance', 'Imbalance', '⚖'], ['build', 'Sharding', '▦'], ['restart', 'Restart', '↺'], ['accounting', 'Accounting', 'Σ'], ['scaling', 'Scaling', '↗'], ['deep', 'Deep', '◇'], ['method', 'Method', '❯_']];

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function MenuBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="menubar" role="banner">
      <span className="apple" aria-hidden="true">◆</span>
      <a className="app" href="#top">Distributed Training Lab</a>
      <a className="m" href={blob('SPEC.md')}>Spec</a>
      <a className="m" href={blob('level-1-learn/README.md')}>Learn</a>
      <a className="m" href={blob('level-2-build/README.md')}>Build</a>
      <a className="m" href={blob('docs/research-review.md')}>Research review</a>
      <span className="spacer" />
      <span className="status"><i /> CPU simulation, no GPU claimed</span>
      <a href={REPO}>GitHub ↗</a>
      <span className="clock">{now.toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</span>
    </div>
  );
}

function Dock() {
  const [active, setActive] = useState('top');
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <nav className="dock" aria-label="Sections">
      {SECTIONS.map(([id, label, icon]) => (
        <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined}>
          <span className="ico" aria-hidden="true">{icon}</span><span className="lbl">{label}</span>
        </a>
      ))}
    </nav>
  );
}

function Range({ label, value, min, max, step, onChange, show }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span>{show}</span></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

function Seg({ label, value, options, onChange }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([k, l]) => <button key={k} aria-pressed={value === k} onClick={() => onChange(k)}>{l}</button>)}
    </div>
  );
}

// Small SVG line chart. series: [{ ys, color, width, opacity, dash }]; log scale when lo > 0. Zeros clamp to the floor.
function Lines({ series, label, lo, hi, log = false, ticks, xLabel, W = 460, H = 220, marker }) {
  const pad = { l: 52, r: 10, t: 10, b: 26 };
  const n = Math.max(...series.map((s) => s.ys.length));
  const tf = (v) => (log ? Math.log10(Math.max(lo, v)) : v);
  const y = (v) => pad.t + (1 - (Math.min(tf(hi), Math.max(tf(lo), tf(v))) - tf(lo)) / (tf(hi) - tf(lo))) * (H - pad.t - pad.b);
  const x = (i) => pad.l + (n > 1 ? i / (n - 1) : 0.5) * (W - pad.l - pad.r);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {ticks.map(([v, t]) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} /><text x={pad.l - 6} y={y(v) + 4} textAnchor="end">{t}</text></g>)}
      {series.map((s, k) => <polyline key={k} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeOpacity={s.opacity ?? 1} strokeDasharray={s.dash} points={s.ys.map((v, i) => `${x(i)},${y(v)}`).join(' ')} />)}
      {marker != null && marker < n && <line x1={x(marker)} x2={x(marker)} y1={pad.t} y2={H - pad.b} stroke="var(--warn)" strokeDasharray="4 4" />}
      <text x={pad.l} y={H - 6}>step 1</text><text x={W - pad.r} y={H - 6} textAnchor="end">{xLabel ?? `step ${n}`}</text>
    </svg>
  );
}

export default function App() {
  useReveal();
  return (
    <div className="wrap">
      <MenuBar />
      <div className="glow" aria-hidden="true" />
      <Hero />
      <Learn />
      <Imbalance />
      <Build />
      <Restart />
      <Accounting />
      <Scaling />
      <Deep />
      <Method />
      <Dock />
      <footer>
        Code: Apache-2.0. Recorded numbers on this page are read from <code>level-*/results*.json</code> at build time. The trainers run in your browser as JavaScript ports of the repo’s Python, checked against those files before every build.
      </footer>
    </div>
  );
}

function Hero() {
  return (
    <header className="card hero window" id="top" data-title="distributed-training-lab — README.md">
      <div className="head">
        <p className="eyebrow">Learn · Build · Deep prep</p>
        <span className="pill warn">● simulated ranks in one process, CPU, no GPU number claimed</span>
      </div>
      <h1>Split the training, keep the answer: the averaged gradient matches one process to <em>{e2(LS.max_grad_diff_vs_single_process)}</em>.</h1>
      <p className="sub">
        This lab teaches how data-parallel and sharded training work by simulating the ranks in plain Python, then checks that only the bookkeeping changes.
        Sharded runs restarted from a step-{FS.checkpoint_step} checkpoint land on exactly the same parameters ({FS.restart_max_diff_vs_uninterrupted.toFixed(1)} difference, both strategies).
        Every trainer below runs live in your browser. The real multi-GPU runs the spec asks for have not happened yet, and this page says so wherever it matters.
      </p>
      <div className="actions">
        <a className="btn primary" href="#learn">Run the data-parallel trainer</a>
        <a className="btn ghost" href="#method">What is and isn’t measured</a>
        <span className="hint">Reproduce with <code>python3 level-1-learn/run_learn.py</code></span>
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{f6(LS.final_loss)}</div><div className="cap">Learn loss after {LS.steps} steps, from {LS.initial_loss}</div></div>
        <div className="tile"><div className="num">{e2(LS.max_grad_diff_vs_single_process)}</div><div className="cap">Largest all-reduce vs one-process gradient gap, {LS.num_ranks} ranks</div></div>
        <div className="tile"><div className="num">{f6(FS.final_loss)}</div><div className="cap">Build loss after {FS.steps} steps, from {f6(FS.initial_loss)}</div></div>
        <div className="tile"><div className="num">{FS.restart_max_diff_vs_uninterrupted.toFixed(1)} / {ZS.restart_max_diff_vs_uninterrupted.toFixed(1)}</div><div className="cap">Restart difference, FSDP-class / ZeRO-class</div></div>
      </div>
      <p className="hint" style={{ marginTop: 14 }}>
        CPU equivalence check (Deep prep): <span className={`pill ${deep.all_pass ? 'good' : 'bad'}`}>all_pass {String(deep.all_pass)}</span>{' '}
        <span className="pill info">gpu_numbers_measured {String(deep.gpu_numbers_measured)}</span>
      </p>
    </header>
  );
}

function Learn() {
  const [ranks, setRanks] = useState(LS.num_ranks);
  const [lr, setLr] = useState(LS.learning_rate);
  const [steps, setSteps] = useState(LS.steps);
  const run = useMemo(() => trainDataParallel({ ranks, lr, steps }), [ranks, lr, steps]);
  const isDefault = ranks === LS.num_ranks && lr === LS.learning_rate && steps === LS.steps;
  const equal = new Set(run.shardSizes).size === 1;
  const maxLoss = Math.max(...run.log.flatMap((e) => [e.combined, ...e.rankLosses]));
  const hiL = 10 ** Math.ceil(Math.log10(Math.max(maxLoss, 1e-3)));
  const lossTicks = [-4, -3, -2, -1, 0, 1, 2, 3, 4].map((k) => 10 ** k).filter((v) => v >= 1e-4 && v <= hiL).map((v) => [v, v >= 1 ? String(v) : v.toExponential(0)]);
  const diffs = run.log.map((e) => e.gradDiff);
  const maxDiff = Math.max(...diffs);
  const diffHi = maxDiff > 1e-6 ? 10 : 1e-14;
  return (
    <section id="learn" className="card window reveal" data-title="data_parallel.py — live trainer">
      <div className="head">
        <div>
          <p className="eyebrow">Learn · data parallel, computed live</p>
          <h2>Each rank sees part of the data. The averaged gradient is <em>the same</em> gradient.</h2>
          <p className="sub">
            A 3-parameter linear model on {LS.num_examples} examples. Each simulated rank computes the gradient on its own shard, an all-reduce averages them, and every replica applies the same update.
            The right chart shows how far that average sits from the single-process full-batch gradient at each step. With equal shards the gap is float rounding only, a few units in the last place.
            Try {sweep.rank_sweep.filter((r) => !r.equal_shards).map((r) => r.num_ranks).join(', ')} ranks: shards become unequal, plain averaging over-weights the small shards, and the gap jumps.
          </p>
        </div>
        <span className={`pill ${isDefault ? 'good' : 'info'}`}>{isDefault ? (run.finalLoss === LS.final_loss ? 'matches results_learn.json exactly' : `within ${e2(Math.abs(run.finalLoss - LS.final_loss))} of results_learn.json`) : 'live, off the recorded settings'}</span>
      </div>
      <div className="controls">
        <Range label="Simulated ranks" value={ranks} min={1} max={LS.num_examples} step={1} onChange={setRanks} show={`${ranks} (shards ${run.shardSizes.join('+')})`} />
        <Range label="Learning rate" value={lr} min={0.01} max={0.12} step={0.01} onChange={(v) => setLr(Math.round(v * 100) / 100)} show={lr.toFixed(2)} />
        <Range label="Steps" value={steps} min={5} max={80} step={1} onChange={setSteps} show={steps} />
      </div>
      <div className="grid2">
        <div>
          <h3>Loss per rank (thin) and on the full batch (thick), log scale</h3>
          <Lines label="Per-rank and combined training loss by step" log lo={1e-4} hi={hiL} ticks={lossTicks}
            series={[...run.shardSizes.map((_, r) => ({ ys: run.log.map((e) => e.rankLosses[r]), color: RANK_COLORS[r % 8], width: 1.2, opacity: 0.7 })), { ys: run.log.map((e) => e.combined), color: 'var(--text)', width: 2.5 }]} />
          <div className="legend">{run.shardSizes.map((s, r) => <span key={r}><i style={{ background: RANK_COLORS[r % 8] }} />rank {r} ({s} ex.)</span>)}<span><i style={{ background: 'var(--text)' }} />full batch</span></div>
        </div>
        <div>
          <h3>All-reduce average minus full-batch gradient (max abs), log scale</h3>
          <Lines label="Gradient difference between all-reduce average and single process by step" log lo={1e-17} hi={diffHi}
            ticks={diffHi > 1 ? [[1e-16, '1e-16'], [1e-8, '1e-8'], [1, '1']] : [[1e-16, '1e-16'], [1e-15, '1e-15']]}
            series={[{ ys: diffs, color: equal ? 'var(--good)' : 'var(--bad)', width: 2 }]} />
          <p className="hint">Exact zeros sit on the floor line. Largest gap this run: <b>{e2(maxDiff)}</b>{equal ? ', float reduction order only.' : ', a real bias from unequal shards.'}</p>
        </div>
      </div>
      <div className="tiles" style={{ gridTemplateColumns: 'repeat(3, minmax(0, 1fr))' }}>
        <div className="tile"><div className="num">{f6(run.finalLoss)}</div><div className="cap">full-batch loss after {steps} steps</div></div>
        <div className="tile"><div className="num">{e2(run.maxReplica)}</div><div className="cap">largest difference between replicas (they stay identical)</div></div>
        <div className="tile"><div className="num mono-num">[{run.finalParams.map((v) => v.toFixed(3)).join(', ')}]</div><div className="cap">final [w1, w2, b], true values [{TRUE_PARAMS.join(', ')}]. Not converged at the recorded {LS.steps} steps.</div></div>
      </div>
    </section>
  );
}

function Imbalance() {
  const [k, setK] = useState(learn.imbalance_experiment.shard_sizes[0]);
  const c = useMemo(() => splitCompare(k), [k]);
  const isRec = k === learn.imbalance_experiment.shard_sizes[0];
  const names = ['∂w1', '∂w2', '∂b'];
  const rows = [['Naive average (each rank weighted equally)', c.naive, 'var(--bad)'], ['Size-weighted average', c.weighted, 'var(--good)'], ['Full-batch gradient (one process)', c.full, 'var(--text)']];
  const lim = Math.ceil(Math.max(...rows.flatMap((r) => r[1].map(Math.abs))) / 5) * 5;
  const W = 460, H = 220, pad = { l: 40, r: 10, t: 10, b: 26 }, gw = (W - pad.l - pad.r) / 3, bw = gw / 4.5;
  const y = (v) => pad.t + ((lim - v) / (2 * lim)) * (H - pad.t - pad.b);
  return (
    <section id="imbalance" className="card window reveal" data-title="imbalance_experiment.json — split explorer">
      <div className="head">
        <div>
          <p className="eyebrow">Shard imbalance, computed live</p>
          <h2>Uneven shards need a <em>weighted</em> average, or the gradient drifts.</h2>
          <p className="sub">
            Two ranks split the {LS.num_examples} examples at the starting point [0, 0, 0]. A plain all-reduce mean gives the small shard as much say as the big one.
            Weighting by shard size recovers the full-batch gradient at every split. The repo recorded one split; the sweep in <a href={blob('level-1-learn/results_sweep.json')}>results_sweep.json</a> now covers all of them.
          </p>
        </div>
        {isRec && <span className="pill info">recorded {learn.imbalance_experiment.shard_sizes.join('|')}: naive {f6(learn.imbalance_experiment.naive_max_abs_diff_vs_full)}, weighted {learn.imbalance_experiment.weighted_max_abs_diff_vs_full.toFixed(1)}</span>}
      </div>
      <div className="controls"><Range label="Split (rank 0 | rank 1)" value={k} min={1} max={LS.num_examples - 1} step={1} onChange={setK} show={`${k} | ${LS.num_examples - k}`} /></div>
      <div className="grid2">
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Gradient components for naive, weighted and full-batch averaging">
          {[-lim, -lim / 2, 0, lim / 2, lim].map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} strokeWidth={t === 0 ? 1.5 : 1} /><text x={pad.l - 6} y={y(t) + 4} textAnchor="end">{t}</text></g>)}
          {names.map((nm, i) => (
            <g key={nm}>
              {rows.map(([, v, col], j) => <rect key={j} x={pad.l + i * gw + bw * (0.5 + j * 1.15)} width={bw} y={Math.min(y(0), y(v[i]))} height={Math.abs(y(v[i]) - y(0))} fill={col} opacity={j === 2 ? 0.75 : 0.9} rx="3" />)}
              <text x={pad.l + i * gw + gw / 2} y={H - 6} textAnchor="middle">{nm}</text>
            </g>
          ))}
        </svg>
        <div>
          <div className="bars">
            {rows.map(([l, v, col]) => <div className="bar-row" key={l}><div className="lab"><span><i className="sw" style={{ background: col }} />{l}</span><b className="mono">[{v.map((x) => x.toFixed(3)).join(', ')}]</b></div></div>)}
          </div>
          <div className="tiles" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div className="tile"><div className="num" style={{ color: c.naiveDiff > 1e-9 ? 'var(--bad)' : 'var(--good)' }}>{c.naiveDiff < 1e-9 ? e2(c.naiveDiff) : f6(c.naiveDiff)}</div><div className="cap">naive error (max abs vs full batch)</div></div>
            <div className="tile"><div className="num" style={{ color: 'var(--good)' }}>{e2(c.weightedDiff)}</div><div className="cap">weighted error</div></div>
          </div>
        </div>
      </div>
    </section>
  );
}

const PHASES = {
  fsdp: [['All-gather parameters', 'Each rank receives the other shards to rebuild the full parameter copy for the forward pass.'], ['Compute gradient', 'Full-batch gradient on the gathered copy.'], ['Reduce-scatter gradient', 'Each rank keeps only the gradient slice for its own shard.'], ['Update own shard', 'Momentum update on the rank’s parameter and velocity shard. The full copy is dropped.']],
  zero: [['Compute gradient', 'Gradient on the current parameters.'], ['Partition gradient', 'Each rank keeps the gradient slice that matches its optimizer shard.'], ['Update own shard', 'Momentum update on the rank’s parameter and velocity shard.'], ['All-gather after update', 'Updated shards are gathered so every rank has the new parameters.']],
};
const BLOCKS = [['W1', 0, 20], ['b1', 20, 25], ['W2', 25, 30], ['b2', 30, 31]];

function Build() {
  const [ranks, setRanks] = useState(FS.num_ranks);
  const [strategy, setStrategy] = useState('fsdp');
  const [phase, setPhase] = useState(-1);
  const [trainer, setTrainer] = useState(() => new ShardedTrainer({ strategy: 'fsdp', ranks: FS.num_ranks, steps: FS.steps }));
  const [, force] = useState(0);
  const parts = partitionIndices(NUM_PARAMS, ranks);
  const owner = Array.from({ length: NUM_PARAMS }, (_, i) => parts.findIndex((p) => p.includes(i)));
  const reset = (r = ranks, s = strategy) => { setTrainer(new ShardedTrainer({ strategy: s, ranks: r, steps: FS.steps })); setPhase(-1); };
  const ph = PHASES[strategy];
  const next = () => {
    if (trainer.step >= FS.steps) return;
    if (phase + 1 >= ph.length) { trainer.runStep(); setPhase(-1); force((x) => x + 1); return; }
    setPhase(phase + 1);
  };
  const runAll = () => { trainer.runTo(FS.steps); setPhase(-1); force((x) => x + 1); };
  const ref = useMemo(() => new ShardedTrainer({ strategy: 'fsdp', ranks: 1, steps: FS.steps }).runTo(FS.steps).losses, []);
  const done = trainer.step >= FS.steps;
  const curLoss = trainer.losses.at(-1);
  return (
    <section id="build" className="card window reveal" data-title="sharding.py — 31 parameters across ranks">
      <div className="head">
        <div>
          <p className="eyebrow">Build · sharded training, computed live</p>
          <h2>No rank holds the whole model, yet the run lands on <em>the same weights</em>.</h2>
          <p className="sub">
            A 2-layer MLP (4 inputs, 5 tanh hidden units, 1 output) has {NUM_PARAMS} parameters. Each rank owns a contiguous slice and the momentum buffer for that slice.
            Step through one training step to see the phase order. FSDP-class and ZeRO-class differ only in when the gather happens: in this repo both run the same update arithmetic, so their losses match each other and the single-process run exactly.
          </p>
        </div>
        <Seg label="Strategy" value={strategy} options={[['fsdp', 'FSDP-class'], ['zero', 'ZeRO-class']]} onChange={(s) => { setStrategy(s); reset(ranks, s); }} />
      </div>
      <div className="controls"><Range label="Simulated ranks" value={ranks} min={1} max={8} step={1} onChange={(r) => { setRanks(r); reset(r); }} show={`${ranks} (shards ${parts.map((p) => p.length).join('+')})`} /></div>
      <h3>Parameter ownership map</h3>
      <div className="strip" role="img" aria-label={`${NUM_PARAMS} parameters colored by owning rank`}>
        {owner.map((r, i) => <span key={i} style={{ background: RANK_COLORS[r % 8] }} className={phase >= 0 && /gather/i.test(ph[phase][0]) ? 'pulse' : ''} title={`param ${i}, rank ${r}`} />)}
      </div>
      <div className="strip-labels">{BLOCKS.map(([n, a, b]) => <span key={n} style={{ flex: b - a }}>{n} <small>{a}–{b - 1}</small></span>)}</div>
      <div className="grid2" style={{ marginTop: 18 }}>
        <div>
          <h3>Phase order, step {Math.min(trainer.step + 1, FS.steps)} of {FS.steps}</h3>
          <ol className="phases">
            {ph.map(([t, d], i) => <li key={t} className={i === phase ? 'on' : i < phase ? 'past' : ''}><b>{t}</b><span className="hint">{d}</span></li>)}
          </ol>
          <div className="actions">
            <button className="btn primary" onClick={next} disabled={done}>{phase + 1 >= ph.length ? 'Finish step' : 'Next phase'}</button>
            <button className="btn ghost" onClick={runAll} disabled={done}>Run to step {FS.steps}</button>
            <button className="btn ghost" onClick={() => reset()}>Reset</button>
          </div>
          <p className="hint">{curLoss != null ? `Loss before step ${trainer.step}: ${f6(curLoss)}.` : 'No step taken yet.'}{done && ` Final loss ${f6(lossM(trainer.fullParams(), trainer.ex))}.`}</p>
        </div>
        <div>
          <h3>Loss by step, this run against one process</h3>
          <Lines label="Sharded training loss by step against the single-process reference" lo={0} hi={4.5} ticks={[[0, '0'], [1, '1'], [2, '2'], [3, '3'], [4, '4']]} xLabel={`step ${FS.steps}`}
            series={[{ ys: ref, color: 'var(--muted)', width: 5, opacity: 0.35 }, { ys: trainer.losses, color: RANK_COLORS[0], width: 2 }]} />
          <div className="legend"><span><i style={{ background: 'var(--muted)' }} />single process</span><span><i style={{ background: RANK_COLORS[0] }} />{ranks}-rank {strategy === 'fsdp' ? 'FSDP' : 'ZeRO'}-class (live)</span></div>
          <p className="hint">Recorded 2-rank result in results_build.json: loss {f6(FS.initial_loss)} to {f6(FS.final_loss)}, difference to single process {FS.max_diff_vs_single_process.toFixed(1)} for both strategies.</p>
        </div>
      </div>
    </section>
  );
}

function Restart() {
  const [ck, setCk] = useState(FS.checkpoint_step);
  const [strategy, setStrategy] = useState('fsdp');
  const r = useMemo(() => restartDemo({ strategy, ckptStep: ck, ranks: FS.num_ranks, steps: FS.steps }), [ck, strategy]);
  const rec = strategy === 'fsdp' ? FS : ZS;
  const resumed = r.straight.losses.map((v, i) => (i < ck ? null : r.resumedLosses[i - ck]));
  return (
    <section id="restart" className="card window reveal" data-title={`ckpt_step${ck}.json — save and restart`}>
      <div className="head">
        <div>
          <p className="eyebrow">Checkpoint and restart, computed live</p>
          <h2>Stop anywhere, rebuild from JSON, and finish <em>bit for bit</em> the same.</h2>
          <p className="sub">
            The browser trains to your checkpoint step, writes the parameter shards, velocity shards and config to a JSON string, builds a fresh trainer from that text, and runs on to step {FS.steps}.
            The momentum buffers matter: drop them and the restarted run would drift.
          </p>
        </div>
        <Seg label="Strategy" value={strategy} options={[['fsdp', 'FSDP-class'], ['zero', 'ZeRO-class']]} onChange={setStrategy} />
      </div>
      <div className="controls"><Range label="Checkpoint at step" value={ck} min={1} max={FS.steps - 1} step={1} onChange={setCk} show={ck} /></div>
      <div className="grid2">
        <div>
          <Lines label="Uninterrupted loss with the restarted run overlaid" lo={0} hi={4.5} ticks={[[0, '0'], [2, '2'], [4, '4']]} marker={ck}
            series={[{ ys: r.straight.losses, color: 'var(--muted)', width: 6, opacity: 0.35 }, { ys: resumed.map((v, i) => v ?? r.straight.losses[i]), color: 'var(--good)', width: 2, dash: '5 3' }]} />
          <div className="legend"><span><i style={{ background: 'var(--muted)' }} />uninterrupted</span><span><i style={{ background: 'var(--good)' }} />restarted after the dashed line</span></div>
        </div>
        <div className="tiles" style={{ gridTemplateColumns: '1fr 1fr', marginTop: 0, alignContent: 'start' }}>
          <div className="tile"><div className="num" style={{ color: r.diff === 0 ? 'var(--good)' : 'var(--bad)' }}>{e2(r.diff)}</div><div className="cap">live: max parameter difference at step {FS.steps}</div></div>
          <div className="tile"><div className="num">{fmt(r.bytes)}</div><div className="cap">characters in this browser checkpoint</div></div>
          <div className="tile"><div className="num">{rec.restart_max_diff_vs_uninterrupted.toFixed(1)}</div><div className="cap"><span className="pill info">recorded</span> Python result, checkpoint at step {rec.checkpoint_step}</div></div>
          <div className="tile"><div className="num">{rec.checkpoint_config_hash.slice(0, 8)}</div><div className="cap"><span className="pill info">recorded</span> SHA-256 config hash prefix the Python restart verifies</div></div>
        </div>
      </div>
    </section>
  );
}

function Accounting() {
  const [P, setP] = useState(NUM_PARAMS);
  const [N, setN] = useState(FS.num_ranks);
  const n = Math.min(N, P);
  const a = accounting(P, n);
  const isRec = P === NUM_PARAMS && n === FS.num_ranks;
  const isLearn = P === LS.communication.num_params && n === LS.num_ranks;
  const Rec = ({ on }) => (on ? <span className="pill info">recorded</span> : null);
  const stages = [['DDP (everything replicated)', a.stage.ddp], ['ZeRO-1 (velocity sharded)', a.stage.zero1], ['ZeRO-2 (+ gradients sharded)', a.stage.zero2], ['ZeRO-3 / FSDP, persistent', a.stage.zero3_fsdp_persistent], ['ZeRO-3 / FSDP, peak with gathered copy', a.stage.zero3_fsdp_peak]];
  const pid = useId();
  return (
    <section id="accounting" className="card window reveal" data-title="accounting.py — communication and memory">
      <div className="head">
        <div>
          <p className="eyebrow">Communication and memory, computed from the repo’s formulas</p>
          <h2>Count what would move and what each rank must hold, <em>before</em> renting a GPU.</h2>
          <p className="sub">
            These are logical counts at {FS.communication.bytes_per_param} bytes per value, straight from <a href={blob('level-2-build/sharding.py')}>sharding.py</a>, <a href={blob('level-1-learn/data_parallel.py')}>data_parallel.py</a> and the new <a href={blob('level-2-build/collectives.py')}>collectives.py</a>.
            Nothing was transferred or measured. Activations, latency and real allocator overhead are not modelled.
          </p>
        </div>
      </div>
      <div className="controls">
        <div className="field"><label htmlFor={pid}>Parameter count P</label><input id={pid} type="number" min={1} max={1000000000} value={P} onChange={(e) => setP(Math.max(1, Math.min(1e9, Math.floor(+e.target.value) || 1)))} /></div>
        <Range label="Ranks N" value={N} min={1} max={16} step={1} onChange={setN} show={n} />
        <div className="field"><span className="hint">Presets</span><div className="actions" style={{ marginTop: 4 }}>
          <button className="btn ghost" onClick={() => { setP(NUM_PARAMS); setN(FS.num_ranks); }}>Build MLP</button>
          <button className="btn ghost" onClick={() => { setP(LS.communication.num_params); setN(LS.num_ranks); }}>Learn model</button>
        </div></div>
      </div>
      <div className="grid2">
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead><tr><th>Quantity (per step, all ranks)</th><th>Formula</th><th>Bytes</th></tr></thead>
            <tbody>
              <tr><td>Sharded all-gather <Rec on={isRec} /></td><td className="mono">Σ (P − s) × 4</td><td>{fmt(a.allGatherB)}</td></tr>
              <tr><td>Sharded reduce-scatter <Rec on={isRec} /></td><td className="mono">Σ s × 4</td><td>{fmt(a.reduceScatterB)}</td></tr>
              <tr><td><b>Sharded total</b> <Rec on={isRec} /></td><td className="mono">sum of both</td><td><b>{fmt(a.shardedPerStepB)}</b></td></tr>
              <tr><td>Over {FS.steps} steps <Rec on={isRec} /></td><td className="mono">× steps</td><td>{fmt(a.shardedPerStepB * FS.steps)}</td></tr>
              <tr><td>Learn all-reduce payload <Rec on={isLearn} /></td><td className="mono">P × 4 × N</td><td>{fmt(a.learnAllReduceB)}{isLearn && ` (× ${LS.steps} = ${fmt(LS.communication.total_bytes_all_ranks)})`}</td></tr>
              <tr><td>Ring all-reduce, sent per rank</td><td className="mono">2(N−1)/N × P × 4</td><td>{fmt(a.ringPerRank * 4)}</td></tr>
            </tbody>
          </table>
          <p className="hint">Shards s: {a.sizes.length > 8 ? `${a.sizes.slice(0, 8).join(', ')}…` : a.sizes.join(', ')}. The repo’s formula charges both strategies the same; real ZeRO-3 also re-gathers parameters for the backward pass.</p>
        </div>
        <div>
          <h3>Model-state values held by the largest rank</h3>
          <div className="bars" style={{ marginTop: 12 }}>
            {stages.map(([l, v], i) => (
              <div className="bar-row" key={l}>
                <div className="lab"><span>{l} {isRec && (i === 0 || i === 4) && <Rec on />}</span><b>{fmt(v)}</b></div>
                <div className="track"><div className="fill" style={{ width: `${(v / a.stage.ddp) * 100}%`, background: i === 0 ? 'var(--muted)' : i === 4 ? 'var(--warn)' : 'var(--primary)' }} /></div>
              </div>
            ))}
          </div>
          <p className="hint" style={{ marginTop: 10 }}>Parameters, gradients and one momentum buffer. Recorded at {FS.num_ranks} ranks: peak {FS.memory.peak_elements_max_rank} against a {FS.memory.baseline_full_elements_per_rank}-value baseline. Stage definitions follow the ZeRO paper (Rajbhandari et al., 2020).</p>
        </div>
      </div>
    </section>
  );
}

function Scaling() {
  const rs = sweep.rank_sweep, ring = scaling.ring_all_reduce;
  const W = 460, H = 220, pad = { l: 52, r: 10, t: 14, b: 26 };
  const lo = 1e-17, hi = 10;
  const y = (v) => pad.t + (1 - (Math.log10(Math.max(lo, v)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - pad.t - pad.b);
  const bw = (W - pad.l - pad.r) / rs.length;
  return (
    <section id="scaling" className="card window reveal" data-title="results_sweep.json — rank sweep and ring all-reduce">
      <div className="head">
        <div>
          <p className="eyebrow">Scaling the rank count · recorded CPU runs</p>
          <h2>Float error does not grow with ranks. <em>Unequal shards</em> are what break the average.</h2>
          <p className="sub">
            The repo now sweeps the Learn run from 1 to 8 simulated ranks, and runs a chunked ring all-reduce (reduce-scatter, then all-gather) over the MLP gradients at 1 to 16 ranks.
            These are replays of the committed result files, not live runs. Rounding gaps stay in the last bits and do not rise steadily with N, because they depend on the order of additions, not the number of ranks.
          </p>
        </div>
        <span className="pill info">recorded</span>
      </div>
      <div className="grid2">
        <div>
          <h3>Learn: largest gradient gap by rank count, log scale</h3>
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Maximum gradient difference against the single process by number of ranks">
            {[[1e-16, '1e-16'], [1e-8, '1e-8'], [1, '1']].map(([v, t]) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} /><text x={pad.l - 6} y={y(v) + 4} textAnchor="end">{t}</text></g>)}
            {rs.map((r, i) => {
              const v = r.max_grad_diff_vs_single_process, yy = y(v);
              return (
                <g key={r.num_ranks}>
                  <rect x={pad.l + i * bw + bw * 0.18} width={bw * 0.64} y={yy} height={H - pad.b - yy} rx="4" fill={r.equal_shards ? 'var(--good)' : 'var(--bad)'} opacity="0.85" />
                  <text x={pad.l + i * bw + bw / 2} y={Math.max(yy - 4, pad.t + 8)} textAnchor="middle">{v < 1e-9 ? e2(v) : v.toFixed(2)}</text>
                  <text x={pad.l + i * bw + bw / 2} y={H - 8} textAnchor="middle">{r.num_ranks}</text>
                </g>
              );
            })}
          </svg>
          <div className="legend"><span><i style={{ background: 'var(--good)' }} />equal shards</span><span><i style={{ background: 'var(--bad)' }} />unequal shards (biased naive mean)</span><span>x axis: ranks</span></div>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <h3>Ring all-reduce over {NUM_PARAMS} gradient values</h3>
          <table>
            <thead><tr><th>Ranks</th><th>Most values sent by one rank</th><th>2(N−1)/N × P</th><th>Ring vs full batch</th><th>Plain sum vs full batch</th></tr></thead>
            <tbody>
              {ring.map((r) => (
                <tr key={r.num_ranks}><td>{r.num_ranks}</td><td>{r.max_elements_sent_by_one_rank}</td><td>{r.ideal_elements_per_rank.toFixed(2)}</td><td>{e2(r.ring_max_abs_diff_vs_full)}</td><td>{e2(r.sequential_max_abs_diff_vs_full)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="hint">Every rank ends with an identical copy (largest difference between copies {e2(Math.max(...ring.map((r) => r.rank_copies_max_diff)))}). The per-rank volume approaches 2P as ranks grow, which is why ring all-reduce scales.</p>
        </div>
      </div>
    </section>
  );
}

function Deep() {
  const scripts = Object.entries(kaggle).map(([p, src]) => ({ file: p.split('/').pop(), purpose: (src.match(/^"""(.*)$/m) || [, ''])[1].replace(/\s*Kaggle only\.?$/, '') }));
  const sp = deep.shard_partition_sums, cr = deep.checkpoint_restart_equality, ge = deep.gradient_all_reduce_equivalence;
  const checks = [
    ['Partition covers every parameter once', sp.covers_each_index_exactly_once, `${sp.num_params} parameters, shards ${sp.shard_sizes.join(' + ')}`],
    ['Shard sums add up to the full sum', sp.shard_sum_abs_diff_vs_full === 0, `${sp.shard_element_sums.map((v) => v.toFixed(1)).join(' + ')} = ${sp.full_vector_sum}`],
    ['All-gather rebuilds the vector', sp.all_gather_reconstruction_max_diff === 0, `max diff ${sp.all_gather_reconstruction_max_diff.toFixed(1)}`],
    ['FSDP-class restart equals uninterrupted', cr.fsdp_restart_max_diff_vs_uninterrupted === 0, `checkpoint ${cr.checkpoint_step} of ${cr.total_steps}, diff ${cr.fsdp_restart_max_diff_vs_uninterrupted.toFixed(1)}`],
    ['ZeRO-class restart equals uninterrupted', cr.zero_restart_max_diff_vs_uninterrupted === 0, `diff ${cr.zero_restart_max_diff_vs_uninterrupted.toFixed(1)}`],
    ['Learn all-reduce matches one process', ge.learn_max_grad_diff_vs_single_process < 1e-12, `max diff ${e2(ge.learn_max_grad_diff_vs_single_process)}`],
    ['No GPU number measured or claimed', deep.gpu_numbers_measured === false, `gpu_numbers_measured ${String(deep.gpu_numbers_measured)}`],
  ];
  return (
    <section id="deep" className="card window reveal" data-title="KAGGLE-RUN.md — Deep exercise set">
      <div className="head">
        <div>
          <p className="eyebrow">Deep · prep package only</p>
          <h2>The GPU exercises are written and <em>waiting</em>. No results exist yet.</h2>
          <p className="sub">
            Deep is defined as a two-GPU exercise set for a free Kaggle notebook. The scripts are ready and pass a syntax check, but none has run, so there are no timings, no scaling table and no bandwidth figures here.
            What did run is a CPU equivalence check that the sharding and restart logic is sound.
          </p>
        </div>
        <span className="pill warn">Deep not closed</span>
      </div>
      <div className="grid2">
        <div>
          <h3>Kaggle scripts (level-3-deep/kaggle)</h3>
          <ul className="scripts">
            {scripts.map((s) => <li key={s.file}><a className="mono" href={blob(`level-3-deep/kaggle/${s.file}`)}>{s.file}</a><span className="hint">{s.purpose}</span><span className="pill warn">not run · compile-checked only</span></li>)}
          </ul>
        </div>
        <div>
          <h3>CPU equivalence checklist (results_deep_prep.json)</h3>
          <ul className="scripts">
            {checks.map(([t, ok, d]) => <li key={t}><b>{t}</b><span className="hint">{d}</span><span className={`pill ${ok ? 'good' : 'bad'}`}>{ok ? 'pass' : 'fail'}</span></li>)}
          </ul>
        </div>
      </div>
    </section>
  );
}

function Method() {
  const rows = [
    ['Learn', 'Data-parallel training of the project’s GPT on 2 GPUs with PyTorch DDP, plus a TensorFlow MirroredStrategy bridge run', `Simulated ranks in one Python process on a ${LS.communication.num_params}-parameter model. Teaches the all-reduce; no DDP, no GPU.`, 'warn'],
    ['Build', 'The same run under real FSDP and DeepSpeed with mid-run checkpoint restart, and NCCL bandwidth tests', `FSDP-class and ZeRO-class emulation on a ${NUM_PARAMS}-parameter MLP with JSON checkpoint restart. No libraries, no bandwidth test.`, 'warn'],
    ['Deep', 'A tokens-per-second and cost scaling table by GPU count, a failure log, and a DDP/FSDP/DeepSpeed write-up', 'Kaggle scripts written, not run. CPU equivalence check passes.', 'bad'],
  ];
  const limits = [
    'No real distributed run exists: no torch.distributed, NCCL or GPU at any level.',
    'FSDP-class and ZeRO-class share the same update arithmetic, so their zero difference holds by construction. They differ only in phase order.',
    'Byte and memory figures are logical counts, not measurements, and ignore activations, latency and allocator overhead.',
    'Toy scale and full batch only: no minibatches, no sampler sharding, no randomness to average over.',
    'The Learn model has not converged at the recorded step count; that run is about equivalence, not accuracy.',
    'Prior-art comparison is still a draft; see NOVELTY.md and the research review.',
  ];
  return (
    <section id="method" className="card window reveal" data-title="SPEC.md — what was asked, what was done">
      <p className="eyebrow">Method and limits</p>
      <h2>What the spec asked for, against <em>what actually ran</em>.</h2>
      <div style={{ overflowX: 'auto', marginTop: 16 }}>
        <table>
          <thead><tr><th>Level</th><th>Spec target</th><th>Done in this repo</th><th>Status</th></tr></thead>
          <tbody>{rows.map(([l, s, d, c]) => <tr key={l}><td><b>{l}</b></td><td>{s}</td><td>{d}</td><td><span className={`pill ${c}`}>{c === 'bad' ? 'not run' : 'simulated'}</span></td></tr>)}</tbody>
        </table>
      </div>
      <div className="grid2" style={{ marginTop: 20 }}>
        <div>
          <h3>Threats and limits</h3>
          <ul className="limits">{limits.map((l) => <li key={l}>{l}</li>)}</ul>
        </div>
        <div id="reproduce">
          <h3>Reproduce</h3>
          <ol className="steps" style={{ marginTop: 10 }}>
            <li><span className="n">1</span><div><b>Run the three levels</b><div className="hint"><code>python3 level-1-learn/run_learn.py && python3 level-2-build/run_build.py && python3 level-3-deep/run_deep_prep.py</code></div></div></li>
            <li><span className="n">2</span><div><b>Run the tests ({testCount} unit tests plus a smoke test, counted from tests/ at build time)</b><div className="hint"><code>python3 -m unittest discover -s tests</code></div></div></li>
            <li><span className="n">3</span><div><b>Check nothing drifted</b><div className="hint"><code>git diff --exit-code -- 'level-*/results*.json'</code></div></div></li>
          </ol>
        </div>
      </div>
      <div className="docs" style={{ marginTop: 20 }}>
        {[['SPEC.md', 'Spec', 'Scope and acceptance criteria'], ['level-1-learn/README.md', 'Learn write-up', 'Data parallel, per-step table'], ['level-2-build/README.md', 'Build write-up', 'Sharding and restart'], ['level-3-deep/KAGGLE-RUN.md', 'Kaggle runbook', 'Exact steps for the GPU run'], ['docs/research-review.md', 'Research review', 'Graded gaps and fixes'], ['NOVELTY.md', 'Novelty', 'Draft, not verified']].map(([p, t, d]) => <a key={p} href={blob(p)}><b>{t}</b><small>{d}</small></a>)}
      </div>
    </section>
  );
}
