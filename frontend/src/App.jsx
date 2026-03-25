import { useState, useRef, useEffect } from 'react';
import './index.css';

// ── Constants ────────────────────────────────────────────────────
const RADIUS = 82;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

const VERDICTS = {
  LEGITIMATE: {
    cls: 'safe',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <polyline points="9 12 11 14 15 10" />
      </svg>
    ),
    title: 'Looks Legitimate',
    label: 'LOW RISK',
    body: 'The embedding profile and structural semantics of this posting align with standard professional listings. No significant fraud indicators were detected by the ensemble model.',
  },
  SUSPICIOUS: {
    cls: 'warn',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
    ),
    title: 'Proceed with Caution',
    label: 'MODERATE RISK',
    body: 'Several unusual linguistic patterns were detected. We recommend verifying the employer through independent channels and never providing sensitive personal data before an official offer.',
  },
  FRAUDULENT: {
    cls: 'danger',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
      </svg>
    ),
    title: 'High Risk Detected',
    label: 'FRAUDULENT',
    body: 'Multiple strong fraud indicators present. The embedding profile closely matches known scam archetypes — wire fraud, fake-check, and credential-harvesting patterns. Do not engage.',
  },
};

// ── Ring Meter ───────────────────────────────────────────────────
function RingMeter({ prob }) {
  const [animated, setAnimated] = useState(false);
  const pct = Math.round(prob * 100);
  const offset = animated
    ? CIRCUMFERENCE - prob * CIRCUMFERENCE
    : CIRCUMFERENCE;

  const color =
    prob > 0.65 ? 'var(--danger)' :
      prob > 0.40 ? 'var(--warn)' :
        'var(--safe)';

  useEffect(() => {
    const t = requestAnimationFrame(() => setAnimated(true));
    return () => cancelAnimationFrame(t);
  }, []);

  return (
    <div className="meter-wrap">
      <svg className="ring-svg" width="196" height="196" viewBox="0 0 200 200">
        {/* Faint tick marks */}
        {Array.from({ length: 36 }).map((_, i) => {
          const angle = (i * 10 - 90) * (Math.PI / 180);
          const isMajor = i % 9 === 0;
          const r1 = RADIUS + 16;
          const r2 = RADIUS + (isMajor ? 22 : 18);
          return (
            <line
              key={i}
              x1={100 + r1 * Math.cos(angle)}
              y1={100 + r1 * Math.sin(angle)}
              x2={100 + r2 * Math.cos(angle)}
              y2={100 + r2 * Math.sin(angle)}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={isMajor ? 1.5 : 0.8}
              strokeLinecap="round"
            />
          );
        })}
        {/* Track */}
        <circle className="ring-track" cx="100" cy="100" r={RADIUS} />
        {/* Glow */}
        <circle
          className="ring-glow"
          cx="100" cy="100" r={RADIUS}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          stroke={color}
          style={{ transition: animated ? 'stroke-dashoffset 1.6s cubic-bezier(0.16,1,0.3,1)' : 'none' }}
        />
        {/* Fill */}
        <circle
          className="ring-fill"
          cx="100" cy="100" r={RADIUS}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          stroke={color}
          style={{ transition: animated ? 'stroke-dashoffset 1.6s cubic-bezier(0.16,1,0.3,1)' : 'none' }}
        />
      </svg>
      <div className="meter-content">
        <CountUp to={pct} style={{ color }} className="ring-label" suffix="%" />
        <span className="ring-sub">risk score</span>
      </div>
    </div>
  );
}

// ── Animated count-up ────────────────────────────────────────────
function CountUp({ to, style, className, suffix = '' }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = null;
    const duration = 1400;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setVal(Math.round(ease * to));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [to]);
  return <span className={className} style={style}>{val}{suffix}</span>;
}

// ── Verdict Card ─────────────────────────────────────────────────
function VerdictCard({ label }) {
  const v = VERDICTS[label] || VERDICTS.LEGITIMATE;
  return (
    <div className={`verdict ${v.cls}`}>
      <div className="verdict-header">
        <span className="verdict-icon">{v.icon}</span>
        <div>
          <div className="verdict-eyebrow">{v.label}</div>
          <div className="verdict-title">{v.title}</div>
        </div>
      </div>
      <p>{v.body}</p>
    </div>
  );
}

// ── Insight Chip ─────────────────────────────────────────────────
function InsightChip({ label, value, accent, wide }) {
  return (
    <div className={`insight-chip${wide ? ' wide' : ''}`}>
      <span className="label">{label}</span>
      <span className="value" style={accent ? { color: 'var(--gold)', fontSize: '0.85rem' } : {}}>
        {value}
      </span>
    </div>
  );
}

// ── Insights Panel ───────────────────────────────────────────────
function Insights({ features }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="insights-section">
      <button type="button" className="insights-toggle" onClick={() => setOpen(o => !o)}>
        <span className="toggle-left">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.5 }}>
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          Model Telemetry
        </span>
        <svg
          className={`chevron ${open ? 'open' : ''}`}
          width="16" height="16" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" strokeWidth="2"
          strokeLinecap="round" strokeLinejoin="round"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {open && (
        <div className="insights-body">
          <InsightChip label="Text Length" value={`${features.text_length.toLocaleString()} chars`} />
          <InsightChip label="Flagged Keywords" value={`${features.scam_keyword_count} detected`} />
          <InsightChip label="Requirements" value={features.has_requirements ? 'Specified' : 'Absent'} />
          <InsightChip
            label="Model Pipeline"
            value="fake_job_ensemble.joblib · all-MiniLM-L6-v2"
            accent
            wide
          />
        </div>
      )}
    </div>
  );
}

// ── Field ────────────────────────────────────────────────────────
function Field({ id, label, optional, children }) {
  return (
    <div className="field">
      <label htmlFor={id}>
        {label}
        {optional && <span className="field-optional">optional</span>}
      </label>
      {children}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────
export default function App() {
  const [form, setForm] = useState({ title: '', description: '', requirements: '' });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const resultRef = useRef(null);

  const handleChange = e =>
    setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  const handleSubmit = async e => {
    e.preventDefault();
    if (!form.title.trim() && !form.description.trim()) {
      setError('Please provide at least a job title or description.');
      return;
    }
    setError('');
    setResult(null);
    setLoading(true);

    try {
      const resp = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    } catch (err) {
      setError(
        err.message.includes('fetch')
          ? 'Cannot reach backend. Is FastAPI running on port 8000?'
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const verdictCls = result ? (VERDICTS[result.label]?.cls || 'safe') : null;

  return (
    <>
      <div className="bg-mesh" />
      <div className="app">
        <div className="container">

          {/* ── Header ── */}
          <header className="header">
            <div className="header-badge">
              <span className="dot" />
              AI Job Intelligence
            </div>
            <h1>Job Authenticity<br />Scanner</h1>
            <p>
              Paste a job posting below. Our ensemble meta-learner extracts
              semantics and structural features to surface fraudulent patterns.
            </p>
          </header>

          {/* ── Input Card ── */}
          <div className="card">
            <div className="card-title">
              <span className="icon-bg">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
              </span>
              Posting Data
            </div>

            <form className="form" onSubmit={handleSubmit}>
              <Field id="title" label="Job Title">
                <input
                  id="title"
                  name="title"
                  type="text"
                  value={form.title}
                  onChange={handleChange}
                  placeholder="e.g. Remote Data Entry Assistant ($40/hr, no experience)"
                  autoComplete="off"
                  spellCheck="false"
                />
              </Field>

              <Field id="description" label="Job Description">
                <textarea
                  id="description"
                  name="description"
                  rows={5}
                  value={form.description}
                  onChange={handleChange}
                  placeholder="Paste the full body text of the job posting here…"
                />
              </Field>

              <Field id="requirements" label="Requirements" optional>
                <textarea
                  id="requirements"
                  name="requirements"
                  rows={3}
                  value={form.requirements}
                  onChange={handleChange}
                  placeholder="Any listed skills, experience, or qualifications…"
                />
              </Field>

              {error && (
                <div className="error-banner" role="alert">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                  {error}
                </div>
              )}

              <button type="submit" className="btn-analyze" disabled={loading}>
                {loading ? (
                  <>
                    <div className="spinner" />
                    <span>Extracting embeddings…</span>
                  </>
                ) : (
                  <>
                    <span>Run Inference</span>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="5" y1="12" x2="19" y2="12" />
                      <polyline points="12 5 19 12 12 19" />
                    </svg>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* ── Results ── */}
          {result && (
            <div className="result-section" ref={resultRef}>
              <div className={`card result-card ${verdictCls}`}>
                <div className="card-title">
                  <span className="icon-bg">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10" />
                      <polyline points="12 6 12 12 16 14" />
                    </svg>
                  </span>
                  Inference Results
                </div>

                <div className="result-grid">
                  <RingMeter prob={result.probability} />
                  <VerdictCard label={result.label} />
                </div>

                <Insights features={result.features} />
              </div>
            </div>
          )}

        </div>

        {/* ── Footer ── */}
        <footer className="footer">
          <span>Powered by</span>
          <code>fake_job_ensemble.joblib</code>
          <span>via FastAPI</span>
        </footer>
      </div>
    </>
  );
}
