import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
  useMemo,
  type CSSProperties,
} from "cursor/canvas";

/**
 * LAYOUT LOCK — see:
 *   runs/optim_pretrain_sonitude_dist_count_1_5m/epoch_progress/PROGRESS_CANVAS_LAYOUT.md
 *   .cursor/rules/promoted-epoch-progress-canvas.mdc
 * Update series/stats only; do not reorder top-level H2 sections or replace SVG dBFS axes
 * with canvas LineChart (negative domains break).
 */

const BEST_EPOCH = 23;
const LAST_EPOCH = 30;
const EPOCHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30];

const TRAIN_LOSS = [-412.7922, -485.0443, -503.5054, -517.8342, -527.5342, -534.8488, -535.5344, -539.0243, -543.9383, -548.5277, -546.5952, -547.2462, -550.2488, -547.8367, -550.9107, -553.5606, -550.6869, -553.6538, -551.1252, -554.8998, -550.3165, -553.0157, -549.9542, -550.6505, -554.4116, -550.0472, -548.9542, -552.5817, -549.1487, -553.084, -556.6237];
const VAL_LOSS = [-480.1825, -500.0543, -517.9863, -538.7707, -554.6489, -553.266, -551.2604, -562.5986, -558.7701, -562.4099, -548.3045, -551.177, -557.9045, -551.3209, -563.44, -565.7718, -572.3831, -557.8261, -562.6828, -531.2825, -572.8146, -555.0091, -565.5026, -572.8506, -569.1811, -561.8218, -555.7351, -550.0046, -558.2616, -563.7735, -563.4157];
const TRAIN_AUDIO = [-413.7121, -485.9364, -504.36, -518.6616, -528.3291, -535.603, -536.2671, -539.7142, -544.6305, -549.1888, -547.2401, -547.8983, -550.8929, -548.4765, -551.4673, -554.1112, -551.2259, -554.2179, -551.6766, -555.4541, -550.8684, -553.5691, -550.4997, -551.1964, -554.9625, -550.5823, -549.5107, -553.1419, -549.7207, -553.6422, -557.1165];
const VAL_AUDIO = [-481.0708, -500.7186, -518.8548, -539.4457, -555.5013, -553.9798, -551.8792, -563.2519, -559.4991, -562.9127, -548.89, -551.9317, -558.3578, -552.0084, -563.8783, -566.4211, -573.0754, -558.3184, -563.1184, -531.9379, -573.3168, -555.518, -566.0336, -573.3273, -569.6978, -562.4499, -556.2423, -550.5625, -558.8164, -564.2829, -563.8585];

const TRAIN_SISDRI = [-0.978, -0.356, 0.629, 0.805, 1.09, 1.082, 1.419, 1.521, 1.428, 1.515, 1.589, 1.538, 1.593, 1.684, 1.751, 1.652, 1.931, 1.852, 1.953, 1.679, 2.012, 1.809, 1.958, 2.04, 2.031, 2.182, 2.243, 2.302, 2.313, 2.18, 2.144];
const VAL_SISDRI = [-0.414, 0.804, 1.053, 0.812, 0.531, 0.755, 1.231, 0.905, 0.884, 1.105, 1.638, 1.476, 1.472, 1.648, 1.34, 1.289, 1.034, 1.623, 1.53, 2.161, 1.184, 1.66, 1.531, 1.138, 1.278, 1.588, 1.609, 1.977, 1.66, 1.646, 1.629];
const TRAIN_SNRI = [5.632, 5.822, 5.998, 6.045, 6.07, 6.094, 6.142, 6.144, 6.106, 6.135, 6.173, 6.161, 6.171, 6.175, 6.176, 6.159, 6.21, 6.184, 6.208, 6.161, 6.223, 6.184, 6.219, 6.215, 6.27, 6.297, 6.306, 6.328, 6.346, 6.313, 6.269];
const VAL_SNRI = [4.592, 4.899, 4.969, 4.87, 4.818, 4.862, 4.924, 4.855, 4.924, 4.874, 5.087, 5.056, 5.007, 5.104, 4.915, 4.938, 4.845, 5.006, 5.021, 5.188, 4.885, 5.111, 5.021, 4.877, 4.945, 5.06, 5.075, 5.211, 5.133, 5.059, 5.066];

const TRAIN_COUNT_ACC = [0.395, 0.416, 0.43, 0.447, 0.454, 0.463, 0.472, 0.486, 0.491, 0.498, 0.512, 0.513, 0.531, 0.526, 0.575, 0.584, 0.601, 0.579, 0.596, 0.592, 0.597, 0.593, 0.602, 0.608, 0.604, 0.614, 0.6, 0.596, 0.588, 0.599, 0.64];
const VAL_COUNT_ACC = [0.368, 0.446, 0.47, 0.476, 0.452, 0.496, 0.501, 0.503, 0.508, 0.59, 0.539, 0.511, 0.65, 0.519, 0.669, 0.529, 0.511, 0.647, 0.677, 0.524, 0.618, 0.632, 0.627, 0.653, 0.627, 0.553, 0.619, 0.603, 0.597, 0.628, 0.683];
const TRAIN_COUNT_LOSS = [1.677, 1.6397, 1.5681, 1.5151, 1.4511, 1.3714, 1.3308, 1.2474, 1.2527, 1.1923, 1.1614, 1.1767, 1.1606, 1.153, 0.9912, 0.9785, 0.9559, 1.0054, 0.9816, 0.9869, 0.9821, 0.986, 0.9699, 0.9714, 0.9817, 0.9505, 0.9931, 1.0007, 1.0233, 0.9959, 0.8681];
const VAL_COUNT_LOSS = [1.6433, 1.2071, 1.6068, 1.2301, 1.575, 1.3059, 1.1215, 1.1895, 1.3371, 0.8963, 1.0573, 1.3902, 0.8017, 1.2589, 0.7713, 1.1803, 1.265, 0.8779, 0.7669, 1.1958, 0.8975, 0.9086, 0.9527, 0.8477, 0.9264, 1.1433, 0.9089, 1.0049, 0.9993, 0.9126, 0.7828];

const TRAIN_DIST_MAE = [0.2387, 0.227, 0.2264, 0.2247, 0.2214, 0.222, 0.2199, 0.2186, 0.2169, 0.2181, 0.2176, 0.2181, 0.2184, 0.2172, 0.2141, 0.216, 0.2166, 0.2157, 0.2155, 0.2151, 0.216, 0.2143, 0.2165, 0.2165, 0.2146, 0.2153, 0.2165, 0.215, 0.2146, 0.2153, 0.2142];
const VAL_DIST_MAE = [0.1752, 0.1752, 0.1756, 0.1692, 0.1713, 0.167, 0.1658, 0.1651, 0.165, 0.1635, 0.1645, 0.1671, 0.1607, 0.1646, 0.1613, 0.1648, 0.1637, 0.1629, 0.1618, 0.1655, 0.1603, 0.1629, 0.1622, 0.1592, 0.1606, 0.1635, 0.1599, 0.1632, 0.1618, 0.159, 0.1594];
const TRAIN_DIST_LOSS = [0.0413, 0.0371, 0.037, 0.0365, 0.0356, 0.036, 0.0355, 0.035, 0.0348, 0.035, 0.0348, 0.0349, 0.0351, 0.0347, 0.0339, 0.0343, 0.0346, 0.034, 0.0341, 0.0341, 0.0346, 0.0339, 0.0344, 0.0344, 0.0339, 0.0343, 0.0344, 0.0341, 0.0338, 0.0342, 0.0339];
const VAL_DIST_LOSS = [0.0288, 0.029, 0.0292, 0.0272, 0.0276, 0.0266, 0.0263, 0.0263, 0.0263, 0.0258, 0.0264, 0.0268, 0.0251, 0.0263, 0.0253, 0.0265, 0.0261, 0.0256, 0.0253, 0.0265, 0.025, 0.0259, 0.0257, 0.0246, 0.0251, 0.0261, 0.0248, 0.0258, 0.0255, 0.0247, 0.0248];

const TRAIN_ACT_LOSS = [0.6072, 0.5371, 0.52, 0.5165, 0.5159, 0.5046, 0.4953, 0.4863, 0.4844, 0.4751, 0.4674, 0.4636, 0.4623, 0.4598, 0.4411, 0.4413, 0.4386, 0.4432, 0.4359, 0.4378, 0.4355, 0.4345, 0.4332, 0.4304, 0.4312, 0.4269, 0.4276, 0.4283, 0.4337, 0.4315, 0.4179];
const VAL_ACT_LOSS = [0.5225, 0.4623, 0.5046, 0.4638, 0.5107, 0.4758, 0.449, 0.4535, 0.4737, 0.4172, 0.4365, 0.4619, 0.3997, 0.4488, 0.4008, 0.4592, 0.4678, 0.4058, 0.3949, 0.4419, 0.4096, 0.4165, 0.4182, 0.4044, 0.4092, 0.4338, 0.4047, 0.4258, 0.4241, 0.4083, 0.3897];

const DIST = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
const INPUT = DIST.map(() => -25.0);
const BEST_02 = [-94.6, -94.5, -94.5, -94.6, -94.6, -94.6, -94.6, -94.6, -94.6, -94.6, -94.6, -94.6, -94.5, -94.5, -94.5, -94.5, -93.6, -37.7, -36.2, -28.1, -37.3, -27.8, -26.7, -27.0, -26.6, -27.2];
const BEST_08 = [-94.5, -94.5, -94.5, -94.6, -94.5, -94.6, -94.6, -94.6, -94.6, -94.5, -94.6, -94.6, -94.6, -94.6, -94.5, -94.5, -94.5, -76.6, -88.5, -52.2, -90.0, -33.9, -36.6, -37.4, -27.6, -26.7];
const BEST_10 = [-94.6, -94.5, -94.6, -94.5, -94.6, -94.6, -94.6, -94.6, -94.6, -94.6, -94.6, -94.5, -94.6, -94.6, -94.6, -94.5, -94.5, -90.3, -86.7, -50.1, -92.0, -34.0, -39.0, -32.1, -28.0, -27.6];
const BEST_16 = [-94.6, -94.5, -94.6, -94.6, -94.6, -94.6, -94.6, -94.6, -94.6, -94.6, -94.5, -94.6, -94.6, -94.5, -94.6, -94.5, -94.5, -93.6, -91.8, -58.3, -91.6, -41.6, -60.8, -37.7, -31.3, -30.4];

const EVOL_DIST = [3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.5];
const EVOL_EP1 = [-81.9, -81.8, -81.7, -77.2, -77.5, -38.4, -28.3];
const EVOL_BEST = [-94.6, -94.6, -94.5, -75.8, -77.7, -40.8, -28.0];
const EVOL_LAST = [-95.1, -95.1, -95.0, -66.4, -72.3, -38.7, -27.3];

type Series = { name: string; color: string; data: number[]; dashed?: boolean };

function Legend(props: { items: { name: string; color: string }[] }) {
  return (
    <Row gap={14} style={{ flexWrap: "wrap" }}>
      {props.items.map((it) => (
        <span key={it.name} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              width: 12,
              height: 12,
              background: it.color,
              display: "inline-block",
              borderRadius: 2,
            }}
          />
          <Text size="small">{it.name}</Text>
        </span>
      ))}
    </Row>
  );
}

function LineSvg(props: {
  xs: number[];
  series: Series[];
  yMin: number;
  yMax: number;
  xLabel: string;
  yLabel: string;
  bestEpoch?: number;
  yTickStep?: number;
  height?: number;
  width?: number;
  xTickEvery?: number;
}) {
  const theme = useHostTheme();
  const {
    xs,
    series,
    yMin,
    yMax,
    xLabel,
    yLabel,
    bestEpoch,
    height = 260,
    width = 980,
    xTickEvery = 1,
  } = props;

  const pad = { l: 58, r: 18, t: 16, b: 46 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];

  const xOf = (x: number) => pad.l + ((x - x0) / Math.max(x1 - x0, 1e-9)) * innerW;
  const yOf = (v: number) => pad.t + ((yMax - v) / (yMax - yMin)) * innerH;

  const nTicks = 6;
  const yTicks = Array.from({ length: nTicks }, (_, i) => yMax - ((yMax - yMin) * i) / (nTicks - 1));
  const grid = theme.stroke.secondary;
  const label = theme.text.secondary;
  const axis = theme.text.primary;

  const poly = (data: number[]) =>
    data.map((v, i) => `${xOf(xs[i]).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" } satisfies CSSProperties}>
      <rect x={pad.l} y={pad.t} width={innerW} height={innerH} fill="none" stroke={grid} strokeWidth={1} />
      {yTicks.map((t, i) => (
        <g key={`yt-${i}`}>
          <line x1={pad.l} x2={pad.l + innerW} y1={yOf(t)} y2={yOf(t)} stroke={grid} strokeWidth={1} opacity={0.45} />
          <text x={pad.l - 8} y={yOf(t) + 4} textAnchor="end" fontSize={11} fill={label}>
            {Math.abs(t) >= 10 ? t.toFixed(0) : t.toFixed(2)}
          </text>
        </g>
      ))}
      {xs.map((x, i) =>
        i % xTickEvery === 0 ? (
          <text key={`xt-${x}`} x={xOf(x)} y={pad.t + innerH + 18} textAnchor="middle" fontSize={11} fill={label}>
            {x}
          </text>
        ) : null,
      )}
      {bestEpoch != null ? (
        <line
          x1={xOf(bestEpoch)}
          x2={xOf(bestEpoch)}
          y1={pad.t}
          y2={pad.t + innerH}
          stroke={axis}
          strokeWidth={1.5}
          strokeDasharray="5 4"
          opacity={0.7}
        />
      ) : null}
      {series.map((s) => (
        <g key={s.name}>
          <polyline
            fill="none"
            stroke={s.color}
            strokeWidth={s.dashed ? 1.5 : 2}
            strokeDasharray={s.dashed ? "5 4" : undefined}
            points={poly(s.data)}
          />
          {s.data.map((v, i) => (
            <circle key={`${s.name}-${i}`} cx={xOf(xs[i])} cy={yOf(v)} r={2.1} fill={s.color} />
          ))}
        </g>
      ))}
      <text x={pad.l + innerW / 2} y={height - 6} textAnchor="middle" fontSize={12} fill={axis}>
        {xLabel}
      </text>
      <text
        x={14}
        y={pad.t + innerH / 2}
        textAnchor="middle"
        fontSize={12}
        fill={axis}
        transform={`rotate(-90 14 ${pad.t + innerH / 2})`}
      >
        {yLabel}
      </text>
    </svg>
  );
}

function DistanceDbfsSvg(props: {
  distances: number[];
  series: Series[];
  yMin?: number;
  yMax?: number;
  bubbleM?: number;
  height?: number;
  width?: number;
  /** RT60 0.2 s passband ripple cursors (max/min + ΔdB inside bubble). */
  ripple?: {
    data: number[];
    color: string;
    bubbleM?: number;
    floorDbfs?: number;
  };
}) {
  const theme = useHostTheme();
  const {
    distances,
    series,
    yMin = -95,
    yMax = -20,
    bubbleM = 1.5,
    height = 300,
    width = 980,
    ripple,
  } = props;
  const pad = { l: 58, r: 18, t: 18, b: 48 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const xMin = Math.max(...distances);
  const xMax = Math.min(...distances);
  const xOf = (d: number) => pad.l + ((xMin - d) / (xMin - xMax)) * innerW;
  const yOf = (v: number) => pad.t + ((yMax - v) / (yMax - yMin)) * innerH;
  const yTicks = [-20, -30, -40, -50, -60, -70, -80, -90];
  const xTicks = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5];
  const grid = theme.stroke.secondary;
  const label = theme.text.secondary;
  const axis = theme.text.primary;
  const poly = (data: number[]) =>
    data.map((v, i) => `${xOf(distances[i]).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");

  const rippleMark = useMemo(() => {
    if (!ripple) return null;
    const bub = ripple.bubbleM ?? bubbleM;
    const floor = ripple.floorDbfs ?? -60;
    const pts = distances
      .map((d, i) => ({ d, v: ripple.data[i], i }))
      .filter((p) => p.d < bub && p.v > floor);
    if (pts.length < 2) return null;
    let hi = pts[0];
    let lo = pts[0];
    for (const p of pts) {
      if (p.v > hi.v) hi = p;
      if (p.v < lo.v) lo = p;
    }
    const delta = hi.v - lo.v;
    return { hi, lo, delta };
  }, [ripple, distances, bubbleM]);

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" } satisfies CSSProperties}>
      <rect x={pad.l} y={pad.t} width={innerW} height={innerH} fill="none" stroke={grid} strokeWidth={1} />
      {yTicks.map((t) => (
        <g key={`y-${t}`}>
          <line x1={pad.l} x2={pad.l + innerW} y1={yOf(t)} y2={yOf(t)} stroke={grid} strokeWidth={1} opacity={0.5} />
          <text x={pad.l - 8} y={yOf(t) + 4} textAnchor="end" fontSize={11} fill={label}>
            {t}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g key={`x-${t}`}>
          <line x1={xOf(t)} x2={xOf(t)} y1={pad.t} y2={pad.t + innerH} stroke={grid} strokeWidth={1} opacity={0.35} />
          <text x={xOf(t)} y={pad.t + innerH + 18} textAnchor="middle" fontSize={11} fill={label}>
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <line x1={xOf(bubbleM)} x2={xOf(bubbleM)} y1={pad.t} y2={pad.t + innerH} stroke={axis} strokeWidth={2} strokeDasharray="6 4" />
      <text x={xOf(bubbleM) + 6} y={pad.t + 14} fontSize={11} fill={axis}>{`bubble ${bubbleM.toFixed(2)} m`}</text>
      {series.map((s) => (
        <g key={s.name}>
          <polyline fill="none" stroke={s.color} strokeWidth={s.dashed ? 1.6 : 2} strokeDasharray={s.dashed ? "5 4" : undefined} points={poly(s.data)} />
          {s.data.map((v, i) => (
            <circle key={`${s.name}-${i}`} cx={xOf(distances[i])} cy={yOf(v)} r={2.2} fill={s.color} />
          ))}
        </g>
      ))}
      {rippleMark && ripple ? (
        <g>
          {/* Horizontal cursors at passband max / min */}
          <line
            x1={xOf(bubbleM)}
            x2={pad.l + innerW}
            y1={yOf(rippleMark.hi.v)}
            y2={yOf(rippleMark.hi.v)}
            stroke={ripple.color}
            strokeWidth={1.4}
            strokeDasharray="3 3"
            opacity={0.9}
          />
          <line
            x1={xOf(bubbleM)}
            x2={pad.l + innerW}
            y1={yOf(rippleMark.lo.v)}
            y2={yOf(rippleMark.lo.v)}
            stroke={ripple.color}
            strokeWidth={1.4}
            strokeDasharray="3 3"
            opacity={0.9}
          />
          {/* Extremum markers on RT60 0.2 curve */}
          <circle cx={xOf(rippleMark.hi.d)} cy={yOf(rippleMark.hi.v)} r={5} fill="none" stroke={ripple.color} strokeWidth={2} />
          <circle cx={xOf(rippleMark.hi.d)} cy={yOf(rippleMark.hi.v)} r={2} fill={ripple.color} />
          <circle cx={xOf(rippleMark.lo.d)} cy={yOf(rippleMark.lo.v)} r={5} fill="none" stroke={ripple.color} strokeWidth={2} />
          <circle cx={xOf(rippleMark.lo.d)} cy={yOf(rippleMark.lo.v)} r={2} fill={ripple.color} />
          {/* Vertical Δ bracket near the near end */}
          <line
            x1={pad.l + innerW - 10}
            x2={pad.l + innerW - 10}
            y1={yOf(rippleMark.hi.v)}
            y2={yOf(rippleMark.lo.v)}
            stroke={ripple.color}
            strokeWidth={1.6}
          />
          <line
            x1={pad.l + innerW - 16}
            x2={pad.l + innerW - 4}
            y1={yOf(rippleMark.hi.v)}
            y2={yOf(rippleMark.hi.v)}
            stroke={ripple.color}
            strokeWidth={1.6}
          />
          <line
            x1={pad.l + innerW - 16}
            x2={pad.l + innerW - 4}
            y1={yOf(rippleMark.lo.v)}
            y2={yOf(rippleMark.lo.v)}
            stroke={ripple.color}
            strokeWidth={1.6}
          />
          <text
            x={pad.l + innerW - 22}
            y={(yOf(rippleMark.hi.v) + yOf(rippleMark.lo.v)) / 2 + 4}
            textAnchor="end"
            fontSize={12}
            fontWeight={600}
            fill={ripple.color}
          >
            {`ripple ${rippleMark.delta.toFixed(1)} dB`}
          </text>
          <text x={xOf(rippleMark.hi.d) + 8} y={yOf(rippleMark.hi.v) - 8} fontSize={10} fill={ripple.color}>
            {`max ${rippleMark.hi.v.toFixed(1)}`}
          </text>
          <text x={xOf(rippleMark.lo.d) + 8} y={yOf(rippleMark.lo.v) + 14} fontSize={10} fill={ripple.color}>
            {`min ${rippleMark.lo.v.toFixed(1)}`}
          </text>
        </g>
      ) : null}
      <text x={pad.l + innerW / 2} y={height - 8} textAnchor="middle" fontSize={12} fill={axis}>
        Source distance (m) — far → near
      </text>
      <text x={16} y={pad.t + innerH / 2} textAnchor="middle" fontSize={12} fill={axis} transform={`rotate(-90 16 ${pad.t + innerH / 2})`}>
        Signal level (dBFS)
      </text>
    </svg>
  );
}

export default function PromotedEpochProgress() {
  const theme = useHostTheme();
  const c = useMemo(
    () => ({
      train: theme.accent.primary,
      val: "#54A24B",
      snri: "#B279A2",
      warn: "#F58518",
      danger: "#E45756",
      muted: "#9ECAE1",
    }),
    [theme],
  );

  const lossSeries: Series[] = [
    { name: "train/loss", color: c.train, data: TRAIN_LOSS },
    { name: "val/loss", color: c.val, data: VAL_LOSS },
  ];
  const audioSeries: Series[] = [
    { name: "train/audio_loss", color: c.train, data: TRAIN_AUDIO },
    { name: "val/audio_loss", color: c.val, data: VAL_AUDIO },
  ];
  const sepSeries: Series[] = [
    { name: "val SI-SDRi", color: c.val, data: VAL_SISDRI },
    { name: "val SNRi", color: c.snri, data: VAL_SNRI },
  ];
  const countAccSeries: Series[] = [
    { name: "train count acc", color: c.train, data: TRAIN_COUNT_ACC },
    { name: "val count acc", color: c.val, data: VAL_COUNT_ACC },
  ];
  const countLossSeries: Series[] = [
    { name: "train count CE", color: c.train, data: TRAIN_COUNT_LOSS },
    { name: "val count CE", color: c.val, data: VAL_COUNT_LOSS },
  ];
  const distMaeSeries: Series[] = [
    { name: "train dist MAE (m)", color: c.train, data: TRAIN_DIST_MAE },
    { name: "val dist MAE (m)", color: c.val, data: VAL_DIST_MAE },
  ];
  const distLossSeries: Series[] = [
    { name: "train dist SmoothL1", color: c.train, data: TRAIN_DIST_LOSS },
    { name: "val dist SmoothL1", color: c.val, data: VAL_DIST_LOSS },
  ];
  const actSeries: Series[] = [
    { name: "train activity BCE", color: c.train, data: TRAIN_ACT_LOSS },
    { name: "val activity BCE", color: c.val, data: VAL_ACT_LOSS },
  ];

  const bestDbfs: Series[] = [
    { name: "Input mic0 (−25 dBFS)", color: c.train, data: INPUT },
    { name: "RT60 0.2 s", color: c.warn, data: BEST_02 },
    { name: "RT60 0.8 s", color: c.val, data: BEST_08 },
    { name: "RT60 1.0 s", color: c.snri, data: BEST_10 },
    { name: "RT60 1.6 s", color: c.danger, data: BEST_16 },
  ];
  const evol: Series[] = [
    { name: "epoch 1 mean", color: c.muted, data: EVOL_EP1 },
    { name: `best ep${BEST_EPOCH} mean`, color: c.warn, data: EVOL_BEST },
    { name: `epoch ${LAST_EPOCH} mean`, color: c.danger, data: EVOL_LAST },
    { name: "Input −25 dBFS", color: c.train, data: EVOL_DIST.map(() => -25), dashed: true },
  ];

  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 1100 }}>
      <Stack gap={8}>
        <H1>optim_pretrain_sonitude_dist_count_1_5m — epoch progress</H1>
        <Text tone="secondary">
          Losses, head metrics, and distance→dBFS characterization from checkpoint metric_values
          (epochs 0–{LAST_EPOCH}). Best checkpoint epoch {BEST_EPOCH}; last.pt current_epoch 31.
          Dashed vertical = best epoch.
        </Text>
        <Row gap={8} style={{ flexWrap: "wrap" }}>
          <Pill tone="info">best ep {BEST_EPOCH}</Pill>
          <Pill tone="neutral">last logged ep {LAST_EPOCH}</Pill>
          <Pill tone="warning">passband ripple 0.2s ≈ 11.1 dB</Pill>
          <Pill tone="neutral">SVG axes (true dBFS / loss scale)</Pill>
        </Row>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat label="Best epoch" value={String(BEST_EPOCH)} />
        <Stat label="Val SI-SDRi @ best" value="1.14 dB" tone="warning" />
        <Stat label="Val SI-SDRi @ last" value="1.63 dB" tone="info" />
        <Stat label="Val dist MAE @ best" value="0.159 m" />
      </Grid>

      <H2>Training / validation losses</H2>
      <Stack gap={8}>
        <H3>Total loss</H3>
        <Text tone="secondary" size="small">
          More negative is better for this SNRLP-dominated objective · Y: loss
        </Text>
        <Legend items={lossSeries.map((s) => ({ name: s.name, color: s.color }))} />
        <LineSvg xs={EPOCHS} series={lossSeries} yMin={-590} yMax={-400} xLabel="Epoch" yLabel="Loss" bestEpoch={BEST_EPOCH} />
      </Stack>
      <Stack gap={8}>
        <H3>Audio loss</H3>
        <Legend items={audioSeries.map((s) => ({ name: s.name, color: s.color }))} />
        <LineSvg xs={EPOCHS} series={audioSeries} yMin={-590} yMax={-400} xLabel="Epoch" yLabel="Audio loss" bestEpoch={BEST_EPOCH} />
      </Stack>

      <Divider />

      <H2>Separation metrics</H2>
      <Stack gap={8}>
        <Text tone="secondary" size="small">Val SI-SDRi / SNRi (dB) · higher is better</Text>
        <Legend items={sepSeries.map((s) => ({ name: s.name, color: s.color }))} />
        <LineSvg xs={EPOCHS} series={sepSeries} yMin={-0.5} yMax={5.5} xLabel="Epoch" yLabel="dB" bestEpoch={BEST_EPOCH} />
      </Stack>

      <Divider />

      <H2>Count / distance head metrics</H2>
      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H3>Count accuracy</H3>
          <Legend items={countAccSeries.map((s) => ({ name: s.name, color: s.color }))} />
          <LineSvg xs={EPOCHS} series={countAccSeries} yMin={0.3} yMax={0.75} xLabel="Epoch" yLabel="Accuracy" bestEpoch={BEST_EPOCH} height={220} />
        </Stack>
        <Stack gap={8}>
          <H3>Count CE loss</H3>
          <Legend items={countLossSeries.map((s) => ({ name: s.name, color: s.color }))} />
          <LineSvg xs={EPOCHS} series={countLossSeries} yMin={0.7} yMax={1.8} xLabel="Epoch" yLabel="CE loss" bestEpoch={BEST_EPOCH} height={220} />
        </Stack>
        <Stack gap={8}>
          <H3>Speaker distance MAE</H3>
          <Text tone="secondary" size="small">Metres · lower is better</Text>
          <Legend items={distMaeSeries.map((s) => ({ name: s.name, color: s.color }))} />
          <LineSvg xs={EPOCHS} series={distMaeSeries} yMin={0.15} yMax={0.25} xLabel="Epoch" yLabel="MAE (m)" bestEpoch={BEST_EPOCH} height={220} />
        </Stack>
        <Stack gap={8}>
          <H3>Distance SmoothL1</H3>
          <Legend items={distLossSeries.map((s) => ({ name: s.name, color: s.color }))} />
          <LineSvg xs={EPOCHS} series={distLossSeries} yMin={0.02} yMax={0.045} xLabel="Epoch" yLabel="SmoothL1" bestEpoch={BEST_EPOCH} height={220} />
        </Stack>
      </Grid>
      <Stack gap={8}>
        <H3>Activity BCE</H3>
        <Legend items={actSeries.map((s) => ({ name: s.name, color: s.color }))} />
        <LineSvg xs={EPOCHS} series={actSeries} yMin={0.35} yMax={0.65} xLabel="Epoch" yLabel="BCE" bestEpoch={BEST_EPOCH} height={220} />
      </Stack>

      <Divider />

      <H2>Distance → dBFS characterization</H2>
      <Callout tone="info" title="Axis convention">
        X: source distance far→near (3.0→0.5 m). Y: signal level −20…−95 dBFS. Same as official
        LibriSpeech RT60 overlay / ep46-style plots.
      </Callout>
      <Stack gap={10}>
        <H3>Best epoch {BEST_EPOCH} — RT60 overlay</H3>
        <Text tone="secondary" size="small">
          Orange cursors mark RT60 0.2 s passband max/min inside the bubble (levels &gt; −60 dBFS);
          Δ = passband ripple (lower is flatter near 1.5 m).
        </Text>
        <Legend items={bestDbfs.map((s) => ({ name: s.name, color: s.color }))} />
        <DistanceDbfsSvg
          distances={DIST}
          series={bestDbfs}
          ripple={{ data: BEST_02, color: c.warn }}
        />
      </Stack>
      <Stack gap={10}>
        <H3>Mean RT60 evolution</H3>
        <Legend items={evol.map((s) => ({ name: s.name, color: s.color }))} />
        <DistanceDbfsSvg distances={EVOL_DIST} series={evol} />
      </Stack>

      <Card>
        <CardHeader>Snapshot (selected epochs)</CardHeader>
        <CardBody>
          <Table
            headers={["Epoch", "Val loss", "SI-SDRi", "Count acc", "Dist MAE", "Note"]}
            rows={[
              ["0", "-480", "-0.41", "0.37", "0.175 m", "start"],
              ["19", "-531", "2.16", "0.52", "0.166 m", "peak SI-SDRi"],
              ["23 (best)", "-573", "1.14", "0.65", "0.159 m", "best val loss"],
              ["30 (last logged)", "-563", "1.63", "0.68", "0.159 m", "last.pt ep31"],
            ]}
          />
        </CardBody>
      </Card>

      <Text tone="secondary" size="small">
        Source: runs/optim_pretrain_sonitude_dist_count_1_5m/epoch_progress/metrics_by_epoch.csv
        and val_distance_sweep/*.csv
      </Text>
    </Stack>
  );
}
