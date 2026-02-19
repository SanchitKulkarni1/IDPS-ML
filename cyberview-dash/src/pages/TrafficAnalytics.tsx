import { useEffect, useRef, useState } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { getJson } from "@/lib/api";

type FIItem = { feature: string; importance: number };

const STATIC_FEATURE_IMPORTANCE: FIItem[] = [
  { feature: "Service", importance: 0.28 },
  { feature: "Src Bytes", importance: 0.22 },
  { feature: "Duration", importance: 0.18 },
  { feature: "Protocol", importance: 0.15 },
  { feature: "Dst Bytes", importance: 0.09 },
  { feature: "Flag", importance: 0.05 },
  { feature: "Wrong Fragment", importance: 0.02 },
  { feature: "Land", importance: 0.01 },
];

const POLL_MS = 3000;

const ATTACK_COLORS: Record<string, string> = {
  DoS: "hsl(0, 84%, 60%)",
  Probe: "hsl(38, 92%, 50%)",
  U2R: "hsl(280, 70%, 55%)",
  R2L: "hsl(200, 70%, 50%)",
};

const TrafficAnalytics = () => {
  const [featureImportance, setFeatureImportance] = useState<FIItem[]>(STATIC_FEATURE_IMPORTANCE);
  const [summary, setSummary] = useState<any>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchAll = async () => {
      try {
        const [fiResp, sumResp] = await Promise.allSettled([
          getJson("/api/analytics/feature-importance"),
          getJson("/api/incidents/summary"),
        ]);

        if (!mounted) return;

        // Feature importance
        if (fiResp.status === "fulfilled" && Array.isArray(fiResp.value)) {
          const normalized: FIItem[] = fiResp.value.map((x: any) => ({
            feature: String(x.feature ?? x.name ?? x[0]),
            importance: Number(x.importance ?? x.weight ?? 0),
          }));
          setFeatureImportance(normalized.length ? normalized : STATIC_FEATURE_IMPORTANCE);
        }

        // Live incident summary
        if (sumResp.status === "fulfilled" && sumResp.value) {
          setSummary(sumResp.value);
        }
      } catch (err) {
        console.error("TrafficAnalytics fetch error", err);
      }
    };

    fetchAll();
    pollRef.current = window.setInterval(fetchAll, POLL_MS);

    return () => {
      mounted = false;
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  // --- Derived data from incident summary ---
  const attackTypes = summary?.attack_types ?? {};
  const timeline = summary?.timeline ?? [];
  const topSources = summary?.top_sources ?? [];
  const riskLevels = summary?.risk_levels ?? {};
  const actions = summary?.actions ?? {};
  const totalPackets = summary?.total_packets ?? 0;

  // Attack type breakdown cards
  const attackBreakdown = Object.entries(attackTypes).map(([type, count]) => ({
    type,
    count: count as number,
    color: ATTACK_COLORS[type] || "hsl(220, 50%, 60%)",
  })).sort((a, b) => b.count - a.count);

  // Anomaly timeline from DB
  const anomalyTimeline = timeline.map((t: any) => ({
    hour: t.hour ? t.hour.slice(11) + ":00" : "",
    attacks: t.attacks ?? 0,
    normal: t.normal ?? 0,
    total: t.total ?? 0,
  }));

  // Risk distribution for bar chart
  const riskData = Object.entries(riskLevels).map(([level, count]) => ({
    level,
    count: count as number,
    color:
      level === "critical" ? "hsl(0, 84%, 60%)" :
        level === "high" ? "hsl(38, 92%, 50%)" :
          level === "medium" ? "hsl(200, 70%, 50%)" :
            "hsl(142, 71%, 45%)",
  }));

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Traffic Analytics</h1>
        <p className="text-muted-foreground mt-1">
          Deep dive into network patterns and feature analysis
          {totalPackets > 0 && <span className="ml-2 text-xs">• {totalPackets.toLocaleString()} incidents analyzed</span>}
        </p>
      </div>

      {/* Feature Importance */}
      <div className="chart-card">
        <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
        <p className="text-sm text-muted-foreground mb-6">Key features contributing to malicious traffic detection</p>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={featureImportance} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis type="number" stroke="hsl(var(--muted-foreground))" />
            <YAxis dataKey="feature" type="category" stroke="hsl(var(--muted-foreground))" width={160} />
            <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "0.5rem" }} />
            <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
              {featureImportance.map((_entry, index) => (
                <Cell key={`cell-${index}`} fill={`hsl(211, ${100 - index * 8}%, ${50 + index * 3}%)`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Anomaly Timeline */}
        <div className="chart-card">
          <h3 className="text-lg font-semibold mb-4">Anomaly Detection Timeline</h3>
          {anomalyTimeline.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={anomalyTimeline}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="hour" stroke="hsl(var(--muted-foreground))" />
                <YAxis stroke="hsl(var(--muted-foreground))" />
                <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "0.5rem" }} />
                <Legend />
                <Line type="monotone" dataKey="attacks" name="Attacks" stroke="hsl(var(--destructive))" strokeWidth={3} dot={{ fill: "hsl(var(--destructive))", r: 5 }} />
                <Line type="monotone" dataKey="normal" name="Normal" stroke="hsl(var(--success))" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-muted-foreground text-sm text-center py-12">Start streaming to see timeline data</p>
          )}
        </div>

        {/* Risk Level Distribution */}
        <div className="chart-card">
          <h3 className="text-lg font-semibold mb-4">Risk Level Distribution</h3>
          {riskData.length > 0 ? (
            <div className="space-y-4 mt-6">
              {riskData.map((item) => {
                const pct = totalPackets > 0 ? Math.round((item.count / totalPackets) * 100) : 0;
                return (
                  <div key={item.level}>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium capitalize">{item.level}</span>
                      <span className="text-sm text-muted-foreground">
                        {item.count} ({pct}%)
                      </span>
                    </div>
                    <div className="h-3 bg-muted rounded-full overflow-hidden">
                      <div className="h-full transition-all duration-500 rounded-full" style={{ width: `${pct}%`, backgroundColor: item.color }} />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-muted-foreground text-sm text-center py-12">No data yet</p>
          )}
        </div>
      </div>

      {/* Attack Type Breakdown */}
      <div className="chart-card">
        <h3 className="text-lg font-semibold mb-4">Attack Type Distribution</h3>
        {attackBreakdown.length > 0 ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            {attackBreakdown.map((attack) => (
              <div key={attack.type} className="stat-card text-center">
                <p className="text-sm text-muted-foreground mb-2">{attack.type}</p>
                <p className="text-3xl font-bold" style={{ color: attack.color }}>{attack.count}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted-foreground text-sm text-center py-8">No attacks detected yet — start the stream</p>
        )}
      </div>

      {/* Top Source IPs */}
      {topSources.length > 0 && (
        <div className="chart-card">
          <h3 className="text-lg font-semibold mb-4">Top Source IPs</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="pb-3">IP Address</th>
                  <th className="pb-3">Total Packets</th>
                  <th className="pb-3">Attacks</th>
                  <th className="pb-3">Threat Level</th>
                </tr>
              </thead>
              <tbody>
                {topSources.map((src: any) => {
                  const attackPct = src.total > 0 ? (src.attacks / src.total) * 100 : 0;
                  return (
                    <tr key={src.ip} className="border-t border-border/50">
                      <td className="py-2 font-mono">{src.ip}</td>
                      <td className="py-2">{src.total}</td>
                      <td className="py-2">{src.attacks}</td>
                      <td className="py-2">
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-20 bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${attackPct}%`,
                                backgroundColor: attackPct > 70 ? "hsl(0,84%,60%)" : attackPct > 30 ? "hsl(38,92%,50%)" : "hsl(142,71%,45%)",
                              }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground">{attackPct.toFixed(0)}%</span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Actions Summary */}
      {Object.keys(actions).length > 0 && (
        <div className="chart-card">
          <h3 className="text-lg font-semibold mb-4">Prevention Actions Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
            {Object.entries(actions).map(([action, count]) => (
              <div key={action} className="stat-card text-center">
                <p className="text-sm text-muted-foreground mb-2 capitalize">{action}</p>
                <p className={`text-3xl font-bold ${action === "blocked" ? "text-destructive" : ""}`}>{count as number}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TrafficAnalytics;
