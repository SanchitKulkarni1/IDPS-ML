import { useState, useEffect, useRef } from "react";
import { Download, Search, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getJson } from "@/lib/api";

const POLL_MS = 3000;

type IncidentRow = {
  id: number;
  timestamp: string;
  source_ip: string;
  dest_ip: string;
  protocol: string;
  prediction: string;
  attack_type: string;
  confidence: number;
  risk_level: string;
  action_taken: string;
};

const Logs = () => {
  const [logs, setLogs] = useState<IncidentRow[]>([]);
  const [riskFilter, setRiskFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(true);
  const pollRef = useRef<number | null>(null);

  const fetchLogs = async () => {
    try {
      const data = await getJson("/api/prevention/incidents?limit=200");
      if (Array.isArray(data)) {
        setLogs(
          data.map((r: any) => ({
            id: r.id,
            timestamp: r.timestamp ?? "",
            source_ip: r.source_ip ?? "—",
            dest_ip: r.dest_ip ?? "—",
            protocol: (r.protocol ?? "—").toUpperCase(),
            prediction: r.prediction ?? "—",
            attack_type: r.attack_type ?? r.prediction ?? "—",
            confidence: r.confidence ?? 0,
            risk_level: r.risk_level ?? "low",
            action_taken: r.action_taken ?? "none",
          }))
        );
      }
    } catch (err) {
      console.error("Failed to fetch incidents", err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchLogs();
    pollRef.current = window.setInterval(fetchLogs, POLL_MS);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const filteredLogs = logs.filter((log) => {
    const matchesRisk =
      riskFilter === "all" || log.risk_level.toLowerCase() === riskFilter;
    const matchesSearch =
      !searchTerm ||
      log.source_ip.includes(searchTerm) ||
      log.attack_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.prediction.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesRisk && matchesSearch;
  });

  const getRiskBadgeColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case "critical":
        return "bg-destructive text-destructive-foreground";
      case "high":
        return "bg-warning text-warning-foreground";
      case "medium":
        return "bg-primary text-primary-foreground";
      case "low":
        return "bg-success text-success-foreground";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const getActionBadge = (action: string) => {
    if (action === "blocked")
      return "bg-destructive/10 text-destructive border border-destructive/30";
    return "bg-muted/10 text-muted-foreground";
  };

  const exportCsv = () => {
    const headers = ["ID", "Timestamp", "Source IP", "Dest IP", "Protocol", "Prediction", "Attack Type", "Confidence", "Risk", "Action"];
    const rows = filteredLogs.map(l => [
      l.id, l.timestamp, l.source_ip, l.dest_ip, l.protocol,
      l.prediction, l.attack_type, (l.confidence * 100).toFixed(1) + "%",
      l.risk_level, l.action_taken
    ]);
    const csv = [headers, ...rows].map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `incidents_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Logs & Reports</h1>
          <p className="text-muted-foreground mt-1">
            Live incident history from the IPS database
            {!loading && <span className="ml-2 text-xs">({logs.length} incidents)</span>}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={fetchLogs}>
            <RefreshCw className="h-4 w-4 mr-1" /> Refresh
          </Button>
          <Button className="gap-2" onClick={exportCsv}>
            <Download className="h-4 w-4" />
            Export CSV
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="stat-card">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by IP address, attack type, or prediction..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={riskFilter} onValueChange={setRiskFilter}>
            <SelectTrigger className="w-full md:w-48">
              <SelectValue placeholder="Filter by risk" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Risk Levels</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Logs Table */}
      <div className="chart-card overflow-hidden">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-14">ID</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Source IP</TableHead>
                <TableHead>Dest IP</TableHead>
                <TableHead>Prediction</TableHead>
                <TableHead>Attack Type</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Risk</TableHead>
                <TableHead>Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredLogs.map((log) => (
                <TableRow key={log.id} className="hover:bg-muted/50">
                  <TableCell className="font-mono text-xs">{log.id}</TableCell>
                  <TableCell className="font-mono text-sm">{log.timestamp.replace("T", " ").replace("Z", "")}</TableCell>
                  <TableCell className="font-mono">{log.source_ip}</TableCell>
                  <TableCell className="font-mono text-sm">{log.dest_ip}</TableCell>
                  <TableCell>
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${log.prediction.toLowerCase() !== "normal"
                          ? "bg-destructive/10 text-destructive"
                          : "bg-success/10 text-success"
                        }`}
                    >
                      {log.prediction}
                    </span>
                  </TableCell>
                  <TableCell>{log.attack_type}</TableCell>
                  <TableCell>{(log.confidence * 100).toFixed(0)}%</TableCell>
                  <TableCell>
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getRiskBadgeColor(
                        log.risk_level
                      )}`}
                    >
                      {log.risk_level}
                    </span>
                  </TableCell>
                  <TableCell>
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${getActionBadge(log.action_taken)}`}>
                      {log.action_taken}
                    </span>
                  </TableCell>
                </TableRow>
              ))}
              {filteredLogs.length === 0 && !loading && (
                <TableRow>
                  <TableCell colSpan={9} className="text-center text-muted-foreground py-8">
                    {logs.length === 0
                      ? "No incidents recorded yet — start the stream to generate data"
                      : "No incidents match your filters"}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
};

export default Logs;
