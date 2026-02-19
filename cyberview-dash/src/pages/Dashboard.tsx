import { Activity, Shield, AlertTriangle, CheckCircle, Ban } from "lucide-react";
import {
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import StatCard from "@/components/StatCard";
import { useEffect, useRef, useState } from "react";
import { getJson } from "@/lib/api";

const POLL_MS = 2000;

const Dashboard = () => {
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const intervalRef = useRef<number | null>(null);
  const visibleRef = useRef<boolean>(true);

  const fetchData = async () => {
    if (!visibleRef.current) return;
    try {
      const res = await getJson("/api/incidents/summary");
      if (res) setSummary(res);
    } catch (err) {
      console.error("Dashboard fetch failed", err);
    }
    setLoading(false);
  };

  useEffect(() => {
    const handleVisibility = () => {
      visibleRef.current = document.visibilityState === "visible";
    };
    document.addEventListener("visibilitychange", handleVisibility);
    fetchData();
    intervalRef.current = window.setInterval(fetchData, POLL_MS);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibility);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // --- Live data from incidents DB ---
  const totalPackets = summary?.total_packets ?? 0;
  const normalCount = summary?.normal_count ?? 0;
  const attackCount = summary?.attack_count ?? 0;
  const blockedCount = summary?.blocked_count ?? 0;

  const normalPct = totalPackets > 0 ? ((normalCount / totalPackets) * 100).toFixed(1) : "0.0";
  const attackPct = totalPackets > 0 ? ((attackCount / totalPackets) * 100).toFixed(1) : "0.0";

  // Attack type breakdown for pie chart
  const predictionCounts = summary?.prediction_counts ?? {};
  const pieData = Object.entries(predictionCounts).map(([name, value]) => {
    const isNormal = name.toLowerCase() === "normal" || name.toLowerCase() === "benign";
    return {
      name,
      value: value as number,
      color: isNormal
        ? "hsl(142, 71%, 45%)"
        : name === "DoS"
          ? "hsl(0, 84%, 60%)"
          : name === "Probe"
            ? "hsl(38, 92%, 50%)"
            : name === "U2R"
              ? "hsl(280, 70%, 55%)"
              : name === "R2L"
                ? "hsl(200, 70%, 50%)"
                : "hsl(220, 50%, 60%)",
    };
  });

  // Timeline data
  const timeline = summary?.timeline ?? [];
  const lineData = timeline.map((t: any) => ({
    time: t.hour ? t.hour.slice(11) + ":00" : "",
    normal: t.normal ?? 0,
    attacks: t.attacks ?? 0,
  }));

  // Recent incidents as alerts
  const recentIncidents: any[] = summary?.recent ?? [];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Network Overview</h1>
        <p className="text-muted-foreground mt-1">
          Real-time monitoring and threat detection
          {!loading && totalPackets > 0 && <span className="ml-2 text-xs">• {totalPackets.toLocaleString()} packets analyzed</span>}
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <StatCard
          title="Total Packets"
          value={loading ? "…" : totalPackets.toLocaleString()}
          icon={Activity}
        />
        <StatCard
          title="Normal Traffic"
          value={loading ? "…" : `${normalPct}%`}
          icon={CheckCircle}
          variant="success"
        />
        <StatCard
          title="Threats Detected"
          value={loading ? "…" : attackCount.toLocaleString()}
          icon={AlertTriangle}
          variant="danger"
        />
        <StatCard
          title="Attack Rate"
          value={loading ? "…" : `${attackPct}%`}
          icon={Shield}
        />
        <StatCard
          title="Blocked IPs"
          value={loading ? "…" : blockedCount.toLocaleString()}
          icon={Ban}
          variant="danger"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Line Chart — traffic timeline */}
        <div className="chart-card">
          <h3 className="text-lg font-semibold mb-4">Traffic Over Time</h3>
          {lineData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" />
                <YAxis stroke="hsl(var(--muted-foreground))" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "0.5rem",
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="normal" stroke="hsl(var(--success))" strokeWidth={2} />
                <Line type="monotone" dataKey="attacks" stroke="hsl(var(--destructive))" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-muted-foreground text-sm py-12 text-center">Start streaming to generate timeline data</p>
          )}
        </div>

        {/* Pie Chart — prediction distribution */}
        <div className="chart-card">
          <h3 className="text-lg font-semibold mb-4">Traffic Distribution</h3>
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) =>
                    `${name}: ${(percent * 100).toFixed(1)}%`
                  }
                  outerRadius={100}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "0.5rem",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-muted-foreground text-sm py-12 text-center">No predictions yet</p>
          )}
        </div>
      </div>

      {/* Recent Threat Alerts from live incidents */}
      <div className="chart-card">
        <h3 className="text-lg font-semibold mb-4">Recent Threat Alerts</h3>
        <div className="space-y-3">
          {recentIncidents.filter((a: any) => a.prediction?.toLowerCase() !== "normal").length === 0 && (
            <p className="text-muted-foreground text-sm">No recent attack alerts</p>
          )}
          {recentIncidents
            .filter((a: any) => a.prediction?.toLowerCase() !== "normal")
            .slice(0, 8)
            .map((incident: any) => {
              const risk = (incident.risk_level || "medium").toLowerCase();
              const riskColors: Record<string, string> = {
                critical: "border-destructive bg-destructive/5",
                high: "border-warning bg-warning/5",
                medium: "border-primary bg-primary/5",
                low: "border-success bg-success/5",
              };
              return (
                <div
                  key={incident.id}
                  className={`border p-3 rounded-lg ${riskColors[risk] || ""}`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-foreground">
                        {incident.attack_type} detected — {incident.action_taken === "blocked" ? "⛔ IP Blocked" : "⚠️ Logged"}
                      </p>
                      <p className="text-muted-foreground text-xs mt-1">
                        Source: {incident.source_ip ?? "—"} → {incident.dest_ip ?? "—"} •{" "}
                        {incident.timestamp ? new Date(incident.timestamp).toLocaleString() : "—"}
                        {" "} • Confidence: {((incident.confidence ?? 0) * 100).toFixed(0)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className={`text-xs font-bold uppercase ${risk === "critical" ? "text-destructive" :
                          risk === "high" ? "text-warning" : "text-primary"
                        }`}>{risk}</p>
                    </div>
                  </div>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
