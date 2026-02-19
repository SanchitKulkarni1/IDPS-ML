import { useEffect, useRef, useState } from "react";
import { Activity, Play, Pause } from "lucide-react";
import { Button } from "@/components/ui/button";
import { API_BASE } from "@/lib/api";
import { filePost } from "@/lib/api";

// Types
type DetectionEvent = {
  type?: string;
  timestamp?: string;
  source_ip?: string;
  dst_ip?: string;
  bytes?: number | null;
  protocol?: string | null;
  prediction?: string;
  confidence?: number;
  risk_level?: string;
  flag_icon?: string;
  attack_type?: string;
  input?: Record<string, any>;
  total_packets?: number;
  normal_count?: number;
  malicious_count?: number;
  detection_accuracy?: number;
};

const MAX_FEED = 50;

const LiveAnalysis = () => {
  const [running, setRunning] = useState(false);
  const [feed, setFeed] = useState<DetectionEvent[]>([]);
  const [analysis, setAnalysis] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);

  const esRef = useRef<EventSource | null>(null);
  const reconnectDelayRef = useRef<number>(1000);
  const topSrcRef = useRef<Record<string, number>>({});

  const pushToFeed = (evt: DetectionEvent) => {
    setFeed((prev) => [evt, ...prev].slice(0, MAX_FEED));
  };

  

  const computeAnalysisFromFeed = (currentFeed: DetectionEvent[]) => {
    const total = currentFeed.length;
    const suspicious = currentFeed.filter((p) => {
      const pr = (p.prediction || "").toString().toLowerCase();
      return pr.includes("mal") || Boolean(p.input?.malicious) || p.risk_level === "critical" || p.risk_level === "high";
    }).length;

    const topSrcMap: Record<string, number> = {};
    currentFeed.forEach((p) => {
      const s = p.source_ip || p.input?.src_ip || p.input?.source_ip || "unknown";
      topSrcMap[s] = (topSrcMap[s] || 0) + 1;
    });
    const top = Object.entries(topSrcMap).sort((a, b) => b[1] - a[1])[0] || [];

    return { total, suspicious, topSrc: top };
  };

  // map one server prediction object to DetectionEvent
  const mapServerPredictionToEvent = (item: any): DetectionEvent => {
    // server returns: { features: {...}, prediction: { prediction: "Normal", confidence: 0.55, timestamp: ... } }
    const features = item.features ?? item.input ?? {};
    const predObj = item.prediction ?? item; // some endpoints return prediction inline
    const predLabel = predObj?.prediction ?? predObj?.pred ?? String(predObj);
    const conf = Number(predObj?.confidence ?? predObj?.conf ?? 0);

    // determine risk_level (server may also include one in event)
    let risk_level = item.risk_level ?? undefined;
    if (!risk_level) {
      if (conf >= 0.95) risk_level = "critical";
      else if (conf >= 0.85) risk_level = "high";
      else if (conf >= 0.7) risk_level = "medium";
      else if (conf >= 0.5) risk_level = "low";
      else risk_level = "info";
    }

    const isMal = (predLabel || "").toString().toLowerCase().includes("mal") || risk_level === "critical" || risk_level === "high";

    return {
      type: "detection",
      timestamp: predObj?.timestamp ?? new Date().toISOString(),
      source_ip: features.src_ip ?? features.source_ip ?? item.source_ip ?? "—",
      dst_ip: features.dst_ip ?? features.destination_ip ?? item.dst_ip ?? "—",
      bytes: features.src_bytes ?? features.bytes ?? null,
      protocol: (features.protocol ?? item.protocol ?? "—").toString().toUpperCase(),
      prediction: predLabel,
      confidence: conf,
      risk_level,
      flag_icon: item.flag_icon ?? (isMal ? "⚠️" : "✓"),
      attack_type: item.attack_type ?? predLabel,
      input: features
    };
  };

  // ingest an array of server predictions into feed and update stats/analysis
  const ingestPredictions = (predictions: any[]) => {
    if (!Array.isArray(predictions) || predictions.length === 0) return;

    // Map server predictions to DetectionEvent, newest first
    const events: DetectionEvent[] = predictions.map(mapServerPredictionToEvent);

    // Push events into feed (we want newest first)
    // Server likely returned them in capture order (first..last). We'll add last-first so newest is at top.
    for (let i = events.length - 1; i >= 0; --i) {
      pushToFeed(events[i]);
      // update topSrcRef counts
      const src = events[i].source_ip || events[i].input?.src_ip || "unknown";
      topSrcRef.current[src] = (topSrcRef.current[src] || 0) + 1;
    }

    // Update analysis snapshot from the new combined feed
    setAnalysis(computeAnalysisFromFeed([...events, ...feed].slice(0, MAX_FEED)));

    // update stats if server provided a summary - otherwise we estimate from events
    // attempt to compute summary from provided predictions if provided in the same response
    const malCount = events.reduce((acc, e) => acc + ((e.prediction || "").toString().toLowerCase().includes("mal") ? 1 : 0), 0);
    const total = events.length;
    setStats((prev: any) => ({
      total_packets: (prev?.total_packets ?? 0) + total,
      normal_count: (prev?.normal_count ?? 0) + (total - malCount),
      malicious_count: (prev?.malicious_count ?? 0) + malCount,
      detection_accuracy: prev?.detection_accuracy ?? 0
    }));
  };

  // SSE start/stop
  const startStream = () => {
    if (esRef.current) return;
    reconnectDelayRef.current = 1000;
    const url = `${API_BASE.replace(/\/$/, "")}/api/stream`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => {
      console.log("SSE connected");
      reconnectDelayRef.current = 1000;

      // when stream opens, fetch recent_detections to seed the UI immediately
      (async () => {
        try {
          const res = await fetch(`${API_BASE.replace(/\/$/, "")}/api/dashboard/stats`).then(r => r.json());
          const recents = res?.recent_detections ?? res?.recent ?? [];
          if (Array.isArray(recents) && recents.length > 0) {
            // map server recents to DetectionEvent and preload
            const mapped = recents.map((r: any) => {
              // recents might be simple {timestamp, normal, malicious} or full detection objects
              if (r.input || r.prediction) {
                return mapServerPredictionToEvent(r);
              }
              // fallback: try to interpret
              return {
                type: "detection",
                timestamp: r.timestamp ?? new Date().toISOString(),
                source_ip: r.source_ip ?? "—",
                dst_ip: r.dst_ip ?? "—",
                bytes: r.bytes ?? null,
                protocol: (r.protocol ?? "—").toString().toUpperCase(),
                prediction: r.prediction ?? "—",
                confidence: r.confidence ?? 0,
                risk_level: r.risk_level ?? "info",
                flag_icon: r.flag_icon ?? "✓",
                attack_type: r.attack_type ?? r.prediction ?? "—",
                input: r.input ?? {}
              } as DetectionEvent;
            });

            // push mapped into feed (newest first)
            for (let i = mapped.length - 1; i >= 0; --i) {
              pushToFeed(mapped[i]);
              const src = mapped[i].source_ip || "unknown";
              topSrcRef.current[src] = (topSrcRef.current[src] || 0) + 1;
            }
            setAnalysis(computeAnalysisFromFeed(mapped));
          }

          // seed stats if provided
          if (res && (res.total_packets !== undefined || res.counts !== undefined)) {
            if (res.total_packets !== undefined) {
              setStats({
                total_packets: Number(res.total_packets ?? 0),
                normal_count: Number(res.normal_count ?? 0),
                malicious_count: Number(res.malicious_count ?? 0),
                detection_accuracy: Number(res.detection_accuracy ?? 0),
              });
            } else if (res.counts) {
              const normal = res.counts["Normal"] ?? res.counts["normal"] ?? 0;
              const malicious = res.counts["Malicious"] ?? res.counts["malicious"] ?? 0;
              setStats({
                total_packets: normal + malicious,
                normal_count: normal,
                malicious_count: malicious,
                detection_accuracy: res.detection_accuracy ?? 0,
              });
            }
          }
        } catch (err) {
          console.warn("failed to fetch recent_detections on SSE open:", err);
        }
      })();
    };

    es.onmessage = (e: MessageEvent) => {
      if (!e.data || e.data.trim() === "") return;
      try {
        const data = JSON.parse(e.data) as DetectionEvent;
        if (!data) return;

        // stats event from server
        if (data.type === "stats" || (data.total_packets !== undefined && data.total_packets !== null)) {
          setStats({
            total_packets: Number(data.total_packets ?? 0),
            normal_count: Number(data.normal_count ?? 0),
            malicious_count: Number(data.malicious_count ?? 0),
            detection_accuracy: Number(data.detection_accuracy ?? 0),
          });
          return;
        }

        // detection event
        pushToFeed(data);

        // maintain topSrcRef
        const src = data.source_ip || data.input?.src_ip || data.input?.source_ip || "unknown";
        topSrcRef.current[src] = (topSrcRef.current[src] || 0) + 1;
        const topEntries = Object.entries(topSrcRef.current).sort((a,b) => b[1] - a[1]);
        const top = topEntries[0] ?? [];

        // update analysis snapshot
        setAnalysis((prev: any) => {
          const prevSusp = prev?.suspicious ?? 0;
          const isMal = (data.prediction || "").toString().toLowerCase().includes("mal") || data.risk_level === "critical" || data.risk_level === "high";
          return {
            total: Math.min(MAX_FEED, (prev?.total ?? feed.length) + 1),
            suspicious: prevSusp + (isMal ? 1 : 0),
            topSrc: top,
          };
        });

        // update stats snapshot if server not pushing them frequently
        setStats((s: any) => ({
          ...(s || {}),
          total_packets: (s?.total_packets ?? 0) + 1,
          normal_count: (s?.normal_count ?? 0) + ((data.risk_level === "critical" || data.risk_level === "high") ? 0 : 1),
          malicious_count: (s?.malicious_count ?? 0) + ((data.risk_level === "critical" || data.risk_level === "high") ? 1 : 0),
        }));
      } catch (err) {
        console.warn("SSE parse error", err);
      }
    };

    es.onerror = (err) => {
      console.warn("SSE error", err);
      try { es.close(); } catch {}
      esRef.current = null;
      const delay = reconnectDelayRef.current;
      reconnectDelayRef.current = Math.min(reconnectDelayRef.current * 2, 30000);
      setTimeout(() => { if (running) startStream(); }, delay);
    };
  };

  const stopStream = () => {
    if (esRef.current) {
      try { esRef.current.close(); } catch {}
      esRef.current = null;
    }
  };

  useEffect(() => {
    if (running) startStream(); else stopStream();
    return () => stopStream();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  // fetch initial stats once
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE.replace(/\/$/, "")}/api/dashboard/stats`).then(r => r.json());
        if (res) {
          if (res.total_packets !== undefined || res.normal_count !== undefined) {
            setStats({
              total_packets: Number(res.total_packets ?? 0),
              normal_count: Number(res.normal_count ?? 0),
              malicious_count: Number(res.malicious_count ?? 0),
              detection_accuracy: Number(res.detection_accuracy ?? 0),
            });
          } else if (res.counts) {
            const normal = res.counts["Normal"] ?? res.counts["normal"] ?? 0;
            const malicious = res.counts["Malicious"] ?? res.counts["malicious"] ?? 0;
            setStats({
              total_packets: normal + malicious,
              normal_count: normal,
              malicious_count: malicious,
              detection_accuracy: res.detection_accuracy ?? 0,
            });
          } else {
            setStats(res);
          }
        }
      } catch (e) { /* ignore */ }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- UPDATED runAnalysis: call backend, ingest returned predictions ----------
  const runAnalysis = async () => {
    try {
      const res = await fetch(`${API_BASE.replace(/\/$/, "")}/api/analysis/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // default to "lo" or omit interface to let server pick — change if you prefer
        body: JSON.stringify({ interface: "lo", count: 10, timeout: 3 })
      }).then(r => r.json());

      console.log("analysis result", res);

      // if server returned predictions array, ingest them immediately into feed
      if (res?.predictions && Array.isArray(res.predictions)) {
        ingestPredictions(res.predictions);
      }

      // if server returned a summary use that to set analysis/stats (authoritative)
      if (res?.summary) {
        setAnalysis({ total: res.summary.total, suspicious: res.summary.malicious, topSrc: [] });
        setStats((prev: any) => ({
          ...(prev || {}),
          total_packets: (prev?.total_packets ?? 0) + (res.summary.total ?? 0),
          normal_count: (prev?.normal_count ?? 0) + (res.summary.normal ?? 0),
          malicious_count: (prev?.malicious_count ?? 0) + (res.summary.malicious ?? 0),
        }));
      }
    } catch (err) {
      console.error("runAnalysis failed", err);
    }
  };
  // -------------------------------------------------------------------------------

  const rowClassFor = (p: DetectionEvent) => {
    if (p.risk_level === "critical") return "bg-destructive/15";
    if (p.risk_level === "high") return "bg-warning/10";
    if (p.risk_level === "medium") return "bg-primary/5";
    if (p.risk_level === "low") return "bg-muted/5";
    return "";
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Live Network Analysis</h1>
        <p className="text-muted-foreground mt-1">Capture and analyze live detections streamed from the backend.</p>
      </div>

      <div className="flex gap-3">
        <Button onClick={() => setRunning(r => !r)} size="sm">
          {running ? <Pause className="mr-2" /> : <Play className="mr-2" />} {running ? "Stop Stream" : "Start Stream"}
        </Button>
        <Button onClick={runAnalysis} size="sm">Run Analysis</Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="stat-card col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Recent Detections (live)</h2>
          </div>

          <div className="overflow-auto max-h-80">
            <table className="w-full text-sm table-fixed">
              <thead>
                <tr className="text-left">
                  <th className="w-12">ID</th>
                  <th>Src</th>
                  <th>Dst</th>
                  <th>Proto</th>
                  <th className="w-20">Bytes</th>
                  <th className="w-40">Flag</th>
                </tr>
              </thead>
              <tbody>
                {feed.map((p, idx) => (
                  <tr key={`${p.source_ip}-${p.timestamp}-${idx}`} className={rowClassFor(p)}>
                    <td className="font-mono">{idx + 1}</td>
                    <td>{p.source_ip ?? p.input?.src_ip ?? p.input?.source_ip ?? "—"}</td>
                    <td>{p.dst_ip ?? p.input?.dst_ip ?? p.input?.destination_ip ?? "—"}</td>
                    <td>{(p.protocol || p.input?.protocol || "—").toString().toUpperCase()}</td>
                    <td className="font-mono">{p.bytes ?? p.input?.src_bytes ?? "—"}</td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span
                          className={`inline-flex items-center justify-center w-8 h-6 rounded-md text-sm font-semibold ${
                            (p.risk_level === "critical" && "bg-destructive/20 text-destructive border border-destructive") ||
                            (p.risk_level === "high" && "bg-warning/10 text-warning border border-warning") ||
                            (p.risk_level === "medium" && "bg-primary/10 text-primary border border-primary") ||
                            (p.risk_level === "low" && "bg-muted/10 text-muted border border-muted") ||
                            "bg-success/10 text-success border border-success"
                          }`}
                          title={`${p.attack_type ?? p.prediction ?? "—"} • Confidence: ${Number(p.confidence ?? 0).toFixed(2)}`}
                        >
                          {p.flag_icon ?? ((p.prediction || "").toString().toUpperCase().includes("MAL") ? "⚠️" : "✓")}
                        </span>

                        <div className="text-xs text-muted-foreground">
                          <div>{p.attack_type ?? p.prediction ?? "-"}</div>
                          <div className="text-[11px]">{Math.round((Number(p.confidence ?? 0) * 100))}%</div>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
                {feed.length === 0 && (
                  <tr>
                    <td colSpan={6} className="text-muted-foreground p-6 text-center">No live detections yet — start stream</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="stat-card">
          <h3 className="text-sm font-medium mb-2">Analysis Summary</h3>
          {analysis ? (
            <div className="space-y-2">
              <div>Total packets: <strong>{analysis.total}</strong></div>
              <div>Suspicious: <strong>{analysis.suspicious}</strong></div>
              <div>Top source: <strong>{analysis.topSrc?.[0] ?? "—"}</strong> <span className="text-muted-foreground">({analysis.topSrc?.[1] ?? 0})</span></div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Run analysis to see results.</p>
          )}

          {stats && (
            <div className="mt-4 space-y-1 text-sm text-muted-foreground">
              <div>Total Packets (from backend): {stats.total_packets}</div>
              <div>Normal: {stats.normal_count}</div>
              <div>Attack: {stats.malicious_count}</div>
              <div>Accuracy: {stats.detection_accuracy ? `${stats.detection_accuracy.toFixed(2)}%` : "—"}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveAnalysis;
