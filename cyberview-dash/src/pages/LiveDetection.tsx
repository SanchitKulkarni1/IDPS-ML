// src/components/LiveDetection.tsx
import React, { useEffect, useState } from "react";
import { Upload, Radio } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { API_BASE, jsonPost, filePost } from "@/lib/api";

const DEFAULT_PROTOCOLS = ["TCP", "UDP", "ICMP", "UNKNOWN"];
const DEFAULT_SERVICES = ["http", "https", "dns", "ssh", "other"];
const DEFAULT_FLAGS = ["SF", "S0", "REJ", "OTHER"];

type CsvRow = Record<string, any>;

const CLASS_ID_TO_NAME: Record<string, string> = {
  "0": "ddos",
  "1": "u2r",
  "2": "r2l",
  "3": "probe",
  // add other numeric mappings if backend uses different ids
};

const LiveDetection: React.FC = () => {
  const [features, setFeatures] = useState({
    duration: 50,
    protocol: "TCP",
    service: "http",
    flag: "SF",
    srcBytes: 50,
    dstBytes: 50,
    land: 0,
    wrongFragment: 0,
  });

  const [protocolOptions, setProtocolOptions] = useState<string[]>(DEFAULT_PROTOCOLS);
  const [serviceOptions, setServiceOptions] = useState<string[]>(DEFAULT_SERVICES);
  const [flagOptions, setFlagOptions] = useState<string[]>(DEFAULT_FLAGS);

  const [prediction, setPrediction] = useState<{ type: "normal" | "malicious"; confidence: number } | null>(null);
  const [csvResults, setCsvResults] = useState<CsvRow[] | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingCsv, setLoadingCsv] = useState(false);

  // fetch model metadata for selects (optional)
  useEffect(() => {
    (async () => {
      try {
        const url = `${API_BASE.replace(/\/$/, "")}/api/model/metadata`;
        const res = await fetch(url);
        if (!res.ok) return;
        const json = await res.json();
        if (Array.isArray(json.protocols) && json.protocols.length) setProtocolOptions(json.protocols);
        if (Array.isArray(json.services) && json.services.length) setServiceOptions(json.services);
        if (Array.isArray(json.flags) && json.flags.length) setFlagOptions(json.flags);
      } catch (e) {
        console.warn("Could not load model metadata:", e);
      }
    })();
  }, []);

  // ------------- helpers: normalize labels & extract fields -------------
  function normalizeLabelRaw(raw: any): string {
    if (raw === null || raw === undefined) return "unknown";
    let s = String(raw).trim();
    if (!s) return "unknown";
    s = s.toLowerCase();

    // numeric class id -> name
    if (/^\d+$/.test(s) && CLASS_ID_TO_NAME[s]) return CLASS_ID_TO_NAME[s];

    // synonyms -> canonical
    if (s.includes("ddos") || s.includes("dos") || s.includes("smurf") || s.includes("neptune") || s.includes("syn")) return "ddos";
    if (s.includes("u2r") || s.includes("user2root") || s.includes("user-to-root")) return "u2r";
    if (s.includes("r2l") || s.includes("remote-to-local") || s.includes("r->l")) return "r2l";
    if (s.includes("probe") || s.includes("scan") || s.includes("recon") || s.includes("portscan")) return "probe";

    // preserve canonical if already known
    if (["ddos", "u2r", "r2l", "probe", "normal", "benign", "anomaly", "attack", "intrusion"].includes(s)) return s;

    // fallback return lowercased token (safe)
    return s;
  }

  function extractLabel(row: CsvRow): string {
    if (!row) return "unknown";

    // common direct keys
    const directKeys = ["label", "pred", "prediction", "pred_label", "class", "y_pred"];
    for (const k of directKeys) {
      if (row[k] !== undefined && row[k] !== null) return normalizeLabelRaw(row[k]);
    }

    // nested prediction object: { prediction: { prediction: 0, confidence:.. } }
    if (typeof row.prediction === "object" && row.prediction !== null) {
      const sub = row.prediction.prediction ?? row.prediction.pred ?? row.prediction.label ?? row.prediction.class ?? row.prediction;
      if (sub !== undefined && sub !== null) return normalizeLabelRaw(sub);
    }

    // features wrapper
    if (row.features && typeof row.features === "object") {
      for (const k of ["label", "prediction", "pred"]) {
        if (row.features[k] !== undefined && row.features[k] !== null) return normalizeLabelRaw(row.features[k]);
      }
    }

    // arrays of preds/predictions
    if (Array.isArray(row.preds) && row.preds.length) return normalizeLabelRaw(row.preds[0]);
    if (Array.isArray(row.predictions) && row.predictions.length) {
      const p0 = row.predictions[0];
      if (p0?.prediction) return normalizeLabelRaw(p0.prediction);
      if (p0?.label) return normalizeLabelRaw(p0.label);
    }

    // attack/attack_type fallback
    if (row.attack || row.attack_type) return normalizeLabelRaw(row.attack ?? row.attack_type);

    // log unknown shape up to a few times
    try {
      const win = window as any;
      if (win.__loggedUnknownRows == null) win.__loggedUnknownRows = 0;
      if (win.__loggedUnknownRows < 5) {
        console.warn("LiveDetection: unknown csv row shape sample:", row);
        win.__loggedUnknownRows++;
      }
    } catch (e) {}

    return "unknown";
  }

  function extractConfidence(row: CsvRow): number {
    if (!row) return 0;
    // if prediction is object with confidence
    if (row.prediction && typeof row.prediction === "object") {
      const c = row.prediction.confidence ?? row.prediction.conf ?? row.confidence;
      return Number(c ?? 0);
    }
    if (row.confidence !== undefined) return Number(row.confidence);
    if (row.conf !== undefined) return Number(row.conf);
    return 0;
  }

  function extractAttackType(row: CsvRow): string | null {
    if (!row) return null;
    if (row.attack_type) return normalizeLabelRaw(row.attack_type);
    if (row.prediction && typeof row.prediction === "object" && row.prediction.attack_type) return normalizeLabelRaw(row.prediction.attack_type);
    if (row.attack) return normalizeLabelRaw(row.attack);
    if (row.type) return normalizeLabelRaw(row.type);
    if (row.features && typeof row.features === "object") {
      const f = row.features;
      if (f.service) return normalizeLabelRaw(f.service);
      if (f.protocol) return normalizeLabelRaw(f.protocol);
      if (f.attack_type) return normalizeLabelRaw(f.attack_type);
    }
    if (row.service) return normalizeLabelRaw(row.service);
    if (row.protocol) return normalizeLabelRaw(row.protocol);
    return null;
  }

  // ---------------- summary computation ----------------
  type Summary = {
    total: number;
    perLabel: Record<string, number>;
    avgConfidenceOverall: number; // 0..1
    avgConfidencePerLabel: Record<string, number>; // 0..1
    attackTypeCounts: Record<string, number>;
  };

  function summarizeResults(rows: CsvRow[] | null): Summary | null {
    if (!rows || rows.length === 0) return null;
    const total = rows.length;
    const perLabel: Record<string, number> = {};
    const confAcc: Record<string, { sum: number; count: number }> = {};
    const attackTypeCounts: Record<string, number> = {};

    let totalConfSum = 0;
    let totalConfCount = 0;

    for (const r of rows) {
      const label = extractLabel(r) || "unknown";
      perLabel[label] = (perLabel[label] || 0) + 1;

      let confRaw = extractConfidence(r) ?? 0;
      // if backend uses 0..100 convert to 0..1
      if (confRaw > 1) confRaw = confRaw / 100;
      confRaw = Math.max(0, Math.min(1, Number(confRaw) || 0));

      totalConfSum += confRaw;
      totalConfCount += 1;
      confAcc[label] = confAcc[label] || { sum: 0, count: 0 };
      confAcc[label].sum += confRaw;
      confAcc[label].count += 1;

      const at = extractAttackType(r) ?? "unknown";
      attackTypeCounts[at] = (attackTypeCounts[at] || 0) + 1;
    }

    const avgOverall = totalConfCount ? totalConfSum / totalConfCount : 0;
    const avgPerLabel: Record<string, number> = {};
    for (const lbl of Object.keys(perLabel)) {
      const acc = confAcc[lbl];
      avgPerLabel[lbl] = acc && acc.count ? acc.sum / acc.count : 0;
    }

    return { total, perLabel, avgConfidenceOverall: avgOverall, avgConfidencePerLabel: avgPerLabel, attackTypeCounts };
  }

  const summary = summarizeResults(csvResults);

  // ------------------ UI handlers ------------------
  const handleCsvUpload = async (file: File | null) => {
    setCsvError(null);
    setCsvResults(null);
    if (!file) return;
    setLoadingCsv(true);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      const res = await filePost("/api/predict/csv", fd);
      // normalize into array of row objects
      if (Array.isArray(res)) setCsvResults(res);
      else if (res?.predictions && Array.isArray(res.predictions)) setCsvResults(res.predictions);
      else setCsvResults([res]);
    } catch (err: any) {
      console.error("CSV upload failed", err);
      const msg = (err && (err.error || err.message)) ? (err.error || err.message) : String(err);
      setCsvError(msg);
    } finally {
      setLoadingCsv(false);
    }
  };

  const handlePredict = async () => {
    setPrediction(null);
    setLoadingPredict(true);
    try {
      const payload = {
        duration: Number(features.duration),
        protocol: String(features.protocol),
        service: features.service ? String(features.service) : null,
        flag: String(features.flag),
        src_bytes: Number(features.srcBytes),
        dst_bytes: Number(features.dstBytes),
        land: Number(features.land),
        wrong_fragment: Number(features.wrongFragment),
      };

      const res = await jsonPost("/api/predict", payload);
      if (res?.error) {
        console.error("prediction error", res);
        setPrediction(null);
        setLoadingPredict(false);
        return;
      }

      const labelRaw = res.prediction ?? res.pred ?? res.label ?? "unknown";
      const label = normalizeLabelRaw(labelRaw);
      let confRaw = Number(res.confidence ?? res.conf ?? 0);
      if (confRaw > 1) confRaw = confRaw / 100;
      confRaw = Math.max(0, Math.min(1, confRaw || 0));
      setPrediction({ type: label.includes("mal") ? "malicious" : "normal", confidence: Math.round(confRaw * 100) });
    } catch (err) {
      console.error("Prediction failed", err);
      setPrediction(null);
    } finally {
      setLoadingPredict(false);
    }
  };

  // small metric card component
  const MetricCard: React.FC<{ title: string; value: string | number; subtitle?: string }> = ({ title, value, subtitle }) => (
    <div className="stat-card text-center p-3">
      <div className="text-xs text-muted-foreground">{title}</div>
      <div className="text-2xl font-bold">{value}</div>
      {subtitle && <div className="text-xs text-muted-foreground mt-1">{subtitle}</div>}
    </div>
  );

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Live Detection</h1>
        <p className="text-muted-foreground mt-1">Analyze network traffic in real-time</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Manual Input */}
        <div className="stat-card">
          <div className="flex items-center gap-2 mb-6">
            <Radio className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">Manual Feature Input</h2>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Duration</Label>
                <span className="text-sm font-mono text-muted-foreground">{features.duration}</span>
              </div>
              <Slider value={[features.duration]} onValueChange={(v) => setFeatures(p => ({ ...p, duration: v[0] }))} max={1000} step={1} className="w-full" />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Src Bytes</Label>
                <span className="text-sm font-mono text-muted-foreground">{features.srcBytes}</span>
              </div>
              <Slider value={[features.srcBytes]} onValueChange={(v) => setFeatures(p => ({ ...p, srcBytes: v[0] }))} max={65535} step={1} className="w-full" />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Dst Bytes</Label>
                <span className="text-sm font-mono text-muted-foreground">{features.dstBytes}</span>
              </div>
              <Slider value={[features.dstBytes]} onValueChange={(v) => setFeatures(p => ({ ...p, dstBytes: v[0] }))} max={65535} step={1} className="w-full" />
            </div>

            <div>
              <Label className="block text-sm mb-1">Protocol</Label>
              <select value={features.protocol} onChange={(e) => setFeatures(f => ({ ...f, protocol: e.target.value }))} className="w-full p-2 border rounded">
                {protocolOptions.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>

            <div>
              <Label className="block text-sm mb-1">Service</Label>
              <select value={features.service} onChange={(e) => setFeatures(f => ({ ...f, service: e.target.value }))} className="w-full p-2 border rounded">
                {serviceOptions.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>

            <div>
              <Label className="block text-sm mb-1">Flag</Label>
              <select value={features.flag} onChange={(e) => setFeatures(f => ({ ...f, flag: e.target.value }))} className="w-full p-2 border rounded">
                {flagOptions.map(fl => <option key={fl} value={fl}>{fl}</option>)}
              </select>
            </div>

            <Button onClick={handlePredict} className="w-full" size="lg" disabled={loadingPredict}>
              {loadingPredict ? "Predicting…" : "Predict Traffic Type"}
            </Button>

            {prediction && (
              <div className={`p-4 rounded-lg border-2 ${prediction.type === "malicious" ? "border-destructive bg-destructive/10" : "border-success bg-success/10"}`}>
                <p className="text-sm font-medium mb-1">Prediction Result</p>
                <p className={`text-2xl font-bold ${prediction.type === "malicious" ? "text-destructive" : "text-success"}`}>
                  {prediction.type === "malicious" ? "⚠️ MALICIOUS" : "✓ NORMAL"}
                </p>
                <p className="text-sm text-muted-foreground mt-2">Confidence: {prediction.confidence}%</p>
              </div>
            )}
          </div>
        </div>

        {/* CSV Upload & Summary */}
        <div className="stat-card">
          <div className="flex items-center gap-2 mb-6">
            <Upload className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">CSV Batch Upload</h2>
          </div>

          <div className="space-y-4">
            <div className="border-2 border-dashed rounded-lg p-8 text-center">
              <p className="text-sm text-muted-foreground">Upload CSV with columns:</p>
              <code className="text-xs bg-card p-2 rounded block mx-auto">duration,protocol,service,flag,src_bytes,dst_bytes,land,wrong_fragment</code>
            </div>

            <Input type="file" accept=".csv" onChange={(e) => { const f = e.target.files?.[0]; handleCsvUpload(f ?? null); }} />
            <Button className="w-full" size="lg" onClick={() => alert("Select a CSV file above to upload")} disabled={loadingCsv}>
              {loadingCsv ? "Uploading…" : "Process CSV"}
            </Button>

            {csvError && <div className="text-sm text-destructive mt-2">CSV Error: {csvError}</div>}

            {csvResults && (
              <div className="mt-2">
                <p className="text-sm font-medium mb-2">CSV Results (first 5 rows):</p>
                <pre className="text-xs max-h-48 overflow-auto bg-card p-2 rounded">{JSON.stringify(csvResults.slice(0, 5), null, 2)}</pre>
              </div>
            )}

            {summary && (
              <div className="mt-4">
                <h3 className="text-sm font-semibold mb-2">CSV Summary</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                  <MetricCard title="Total rows" value={summary.total} />
                  <MetricCard title="Avg confidence" value={`${Math.round(summary.avgConfidenceOverall * 100)}%`} subtitle="overall" />
                  <MetricCard title="Unique labels" value={Object.keys(summary.perLabel).length} />
                  <MetricCard title="Top attack type" value={Object.entries(summary.attackTypeCounts).sort((a,b)=>b[1]-a[1])[0]?.[0] ?? "—"} subtitle="most common" />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-3 border rounded">
                    <div className="text-sm font-medium mb-2">Label breakdown</div>
                    <ul className="text-sm">
                      {Object.entries(summary.perLabel).sort((a,b)=>b[1]-a[1]).map(([lbl, cnt]) => {
                        const pct = ((cnt / summary.total) * 100).toFixed(1);
                        const avgConf = Math.round((summary.avgConfidencePerLabel[lbl] ?? 0) * 100);
                        return (
                          <li key={lbl} className="flex justify-between gap-4 py-1 border-b last:border-b-0">
                            <div className="truncate">{lbl}</div>
                            <div className="text-right">
                              <div>{cnt} ({pct}%)</div>
                              <div className="text-xs text-muted-foreground">avg {avgConf}%</div>
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                  </div>

                  <div className="p-3 border rounded">
                    <div className="text-sm font-medium mb-2">Attack type breakdown</div>
                    <ul className="text-sm">
                      {Object.entries(summary.attackTypeCounts).sort((a,b)=>b[1]-a[1]).map(([typ, cnt]) => {
                        const pct = ((cnt / summary.total) * 100).toFixed(1);
                        return (
                          <li key={typ} className="flex justify-between py-1 border-b last:border-b-0">
                            <div className="truncate">{typ}</div>
                            <div className="text-right">{cnt} ({pct}%)</div>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                </div>
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveDetection;
