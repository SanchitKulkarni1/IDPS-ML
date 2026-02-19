import { useEffect, useState, useCallback } from "react";
import {
    ShieldBan,
    ShieldCheck,
    ShieldAlert,
    Ban,
    Unlock,
    RefreshCw,
    Settings2,
    Activity,
    AlertTriangle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
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
import StatCard from "@/components/StatCard";
import { getJson, jsonPost } from "@/lib/api";

// Types
type BlockedIP = {
    id: number;
    ip_address: string;
    reason: string;
    attack_type: string;
    blocked_at: string;
    blocked_by: string;
    expires_at: string | null;
    active: number;
};

type PreventionStatus = {
    mode: "simulation" | "live";
    blocked_count: number;
    total_incidents: number;
    active_rules: number;
    total_rules: number;
};

type Incident = {
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

type Rule = {
    id: number;
    rule_name: string;
    enabled: boolean;
    config: Record<string, any>;
};

const POLL_MS = 3000;

const Prevention = () => {
    const [status, setStatus] = useState<PreventionStatus | null>(null);
    const [blockedIPs, setBlockedIPs] = useState<BlockedIP[]>([]);
    const [incidents, setIncidents] = useState<Incident[]>([]);
    const [rules, setRules] = useState<Rule[]>([]);
    const [loading, setLoading] = useState(true);

    // Manual block form
    const [blockIp, setBlockIp] = useState("");
    const [blockReason, setBlockReason] = useState("");
    const [blockDuration, setBlockDuration] = useState("1");
    const [blockLoading, setBlockLoading] = useState(false);

    // Incident filters
    const [incidentFilter, setIncidentFilter] = useState("all");

    // Fetch all data
    const fetchAll = useCallback(async () => {
        try {
            const [statusRes, blockedRes, incidentsRes, rulesRes] =
                await Promise.allSettled([
                    getJson("/api/prevention/status"),
                    getJson("/api/prevention/blocked"),
                    getJson("/api/prevention/incidents", { limit: 20 }),
                    getJson("/api/prevention/rules"),
                ]);
            if (statusRes.status === "fulfilled") setStatus(statusRes.value);
            if (blockedRes.status === "fulfilled" && Array.isArray(blockedRes.value))
                setBlockedIPs(blockedRes.value);
            if (
                incidentsRes.status === "fulfilled" &&
                Array.isArray(incidentsRes.value)
            )
                setIncidents(incidentsRes.value);
            if (rulesRes.status === "fulfilled" && Array.isArray(rulesRes.value))
                setRules(rulesRes.value);
        } catch (err) {
            console.error("Prevention fetch error", err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchAll();
        const interval = setInterval(fetchAll, POLL_MS);
        return () => clearInterval(interval);
    }, [fetchAll]);

    // Toggle mode
    const toggleMode = async () => {
        try {
            const res = await jsonPost("/api/prevention/toggle-mode", {});
            if (res?.mode) setStatus((s) => (s ? { ...s, mode: res.mode } : s));
        } catch (err) {
            console.error("Toggle mode failed", err);
        }
    };

    // Manual block
    const handleBlock = async () => {
        if (!blockIp.trim()) return;
        setBlockLoading(true);
        try {
            await jsonPost("/api/prevention/block", {
                ip: blockIp.trim(),
                reason: blockReason || "Manual block from dashboard",
                duration_hours: parseFloat(blockDuration) || null,
            });
            setBlockIp("");
            setBlockReason("");
            setBlockDuration("1");
            fetchAll();
        } catch (err) {
            console.error("Block failed", err);
        } finally {
            setBlockLoading(false);
        }
    };

    // Unblock
    const handleUnblock = async (ip: string) => {
        try {
            await jsonPost("/api/prevention/unblock", { ip });
            fetchAll();
        } catch (err) {
            console.error("Unblock failed", err);
        }
    };

    // Toggle rule
    const toggleRule = async (ruleName: string, currentEnabled: boolean) => {
        try {
            const res = await fetch(
                `http://localhost:5000/api/prevention/rules`,
                {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ rule_name: ruleName, enabled: !currentEnabled }),
                }
            );
            if (res.ok) fetchAll();
        } catch (err) {
            console.error("Toggle rule failed", err);
        }
    };

    // Filtered incidents
    const filteredIncidents =
        incidentFilter === "all"
            ? incidents
            : incidents.filter(
                (i) =>
                    i.action_taken === incidentFilter ||
                    i.risk_level === incidentFilter
            );

    const isSimulation = status?.mode === "simulation";

    // Risk badge colors
    const riskColor = (risk: string) => {
        switch (risk?.toLowerCase()) {
            case "critical":
                return "bg-destructive/10 text-destructive border-destructive";
            case "high":
                return "bg-warning/10 text-warning border-warning";
            case "medium":
                return "bg-primary/10 text-primary border-primary";
            default:
                return "bg-success/10 text-success border-success";
        }
    };

    const actionBadge = (action: string) => {
        if (action === "blocked" || action === "already_blocked")
            return "bg-destructive/10 text-destructive";
        if (action === "rate_limited") return "bg-warning/10 text-warning";
        if (action === "alerted") return "bg-primary/10 text-primary";
        return "bg-muted text-muted-foreground";
    };

    const friendlyRuleName = (name: string) =>
        name
            .replace(/_/g, " ")
            .replace(/\b\w/g, (c) => c.toUpperCase());

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-foreground">
                    Intrusion Prevention
                </h1>
                <p className="text-muted-foreground mt-1">
                    Active threat prevention, IP blocking, and auto-response management
                </p>
            </div>

            {/* Mode Banner */}
            <div
                className={`p-4 rounded-lg border-2 flex items-center justify-between ${isSimulation
                        ? "border-primary bg-primary/5"
                        : "border-destructive bg-destructive/5"
                    }`}
            >
                <div className="flex items-center gap-3">
                    {isSimulation ? (
                        <ShieldCheck className="h-6 w-6 text-primary" />
                    ) : (
                        <ShieldAlert className="h-6 w-6 text-destructive" />
                    )}
                    <div>
                        <p className="font-semibold text-foreground">
                            {isSimulation ? "Simulation Mode" : "🔴 Live Mode"}
                        </p>
                        <p className="text-sm text-muted-foreground">
                            {isSimulation
                                ? "Blocks are logged but no iptables rules are applied"
                                : "iptables rules are actively applied — real traffic is being blocked"}
                        </p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <Label htmlFor="mode-toggle" className="text-sm">
                        {isSimulation ? "Simulation" : "Live"}
                    </Label>
                    <Switch
                        id="mode-toggle"
                        checked={!isSimulation}
                        onCheckedChange={toggleMode}
                    />
                </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard
                    title="Blocked IPs"
                    value={loading ? "…" : String(status?.blocked_count ?? 0)}
                    icon={Ban}
                    variant="danger"
                />
                <StatCard
                    title="Total Incidents"
                    value={loading ? "…" : String(status?.total_incidents ?? 0)}
                    icon={Activity}
                />
                <StatCard
                    title="Active Rules"
                    value={
                        loading
                            ? "…"
                            : `${status?.active_rules ?? 0}/${status?.total_rules ?? 0}`
                    }
                    icon={Settings2}
                />
                <StatCard
                    title="System Mode"
                    value={loading ? "…" : isSimulation ? "Simulation" : "Live"}
                    icon={isSimulation ? ShieldCheck : ShieldAlert}
                    variant={isSimulation ? "success" : "danger"}
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Blocked IPs Table */}
                <div className="chart-card">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold">Blocked IPs</h3>
                        <Button variant="outline" size="sm" onClick={fetchAll}>
                            <RefreshCw className="h-4 w-4 mr-1" /> Refresh
                        </Button>
                    </div>
                    <div className="overflow-auto max-h-80">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>IP Address</TableHead>
                                    <TableHead>Reason</TableHead>
                                    <TableHead>By</TableHead>
                                    <TableHead>Expires</TableHead>
                                    <TableHead>Action</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {blockedIPs.length === 0 && (
                                    <TableRow>
                                        <TableCell
                                            colSpan={5}
                                            className="text-center text-muted-foreground py-8"
                                        >
                                            No IPs currently blocked
                                        </TableCell>
                                    </TableRow>
                                )}
                                {blockedIPs.map((b) => (
                                    <TableRow key={b.id}>
                                        <TableCell className="font-mono text-sm">
                                            {b.ip_address}
                                        </TableCell>
                                        <TableCell className="text-sm max-w-[180px] truncate">
                                            {b.reason}
                                        </TableCell>
                                        <TableCell>
                                            <span
                                                className={`text-xs px-2 py-0.5 rounded-full ${b.blocked_by === "auto"
                                                        ? "bg-primary/10 text-primary"
                                                        : "bg-warning/10 text-warning"
                                                    }`}
                                            >
                                                {b.blocked_by}
                                            </span>
                                        </TableCell>
                                        <TableCell className="text-sm">
                                            {b.expires_at
                                                ? new Date(b.expires_at).toLocaleString()
                                                : "Permanent"}
                                        </TableCell>
                                        <TableCell>
                                            <Button
                                                variant="destructive"
                                                size="sm"
                                                onClick={() => handleUnblock(b.ip_address)}
                                            >
                                                <Unlock className="h-3 w-3 mr-1" /> Unblock
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </div>
                </div>

                {/* Manual Block Form */}
                <div className="stat-card">
                    <div className="flex items-center gap-2 mb-6">
                        <ShieldBan className="h-5 w-5 text-destructive" />
                        <h3 className="text-lg font-semibold">Manual IP Block</h3>
                    </div>
                    <div className="space-y-4">
                        <div className="space-y-2">
                            <Label>IP Address</Label>
                            <Input
                                placeholder="e.g. 192.168.1.100"
                                value={blockIp}
                                onChange={(e) => setBlockIp(e.target.value)}
                            />
                        </div>
                        <div className="space-y-2">
                            <Label>Reason</Label>
                            <Input
                                placeholder="Reason for blocking"
                                value={blockReason}
                                onChange={(e) => setBlockReason(e.target.value)}
                            />
                        </div>
                        <div className="space-y-2">
                            <Label>Duration (hours)</Label>
                            <Select value={blockDuration} onValueChange={setBlockDuration}>
                                <SelectTrigger>
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="0.5">30 minutes</SelectItem>
                                    <SelectItem value="1">1 hour</SelectItem>
                                    <SelectItem value="6">6 hours</SelectItem>
                                    <SelectItem value="24">24 hours</SelectItem>
                                    <SelectItem value="168">7 days</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <Button
                            className="w-full"
                            variant="destructive"
                            size="lg"
                            onClick={handleBlock}
                            disabled={blockLoading || !blockIp.trim()}
                        >
                            <Ban className="h-4 w-4 mr-2" />
                            {blockLoading ? "Blocking…" : "Block IP"}
                        </Button>
                    </div>
                </div>
            </div>

            {/* Auto-Response Rules */}
            <div className="chart-card">
                <h3 className="text-lg font-semibold mb-4">Auto-Response Rules</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {rules.map((rule) => (
                        <div
                            key={rule.rule_name}
                            className={`p-4 rounded-lg border ${rule.enabled
                                    ? "border-primary/30 bg-primary/5"
                                    : "border-muted bg-muted/30 opacity-60"
                                }`}
                        >
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="font-medium text-sm">
                                    {friendlyRuleName(rule.rule_name)}
                                </h4>
                                <Switch
                                    checked={rule.enabled}
                                    onCheckedChange={() =>
                                        toggleRule(rule.rule_name, rule.enabled)
                                    }
                                />
                            </div>
                            <p className="text-xs text-muted-foreground mb-2">
                                {rule.config?.description || "No description"}
                            </p>
                            <div className="flex flex-wrap gap-2 text-xs">
                                {rule.config?.min_confidence && (
                                    <span className="px-2 py-0.5 bg-card rounded border">
                                        Confidence ≥ {(rule.config.min_confidence * 100).toFixed(0)}%
                                    </span>
                                )}
                                {rule.config?.block_duration_hours && (
                                    <span className="px-2 py-0.5 bg-card rounded border">
                                        Block: {rule.config.block_duration_hours}h
                                    </span>
                                )}
                                {rule.config?.max_incidents && (
                                    <span className="px-2 py-0.5 bg-card rounded border">
                                        Max: {rule.config.max_incidents} incidents
                                    </span>
                                )}
                                {rule.config?.window_seconds && (
                                    <span className="px-2 py-0.5 bg-card rounded border">
                                        Window: {rule.config.window_seconds}s
                                    </span>
                                )}
                                {rule.config?.max_connections && (
                                    <span className="px-2 py-0.5 bg-card rounded border">
                                        Max conns: {rule.config.max_connections}
                                    </span>
                                )}
                                {rule.config?.attack_types && (
                                    <span className="px-2 py-0.5 bg-card rounded border">
                                        Types: {rule.config.attack_types.join(", ")}
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Recent Incidents */}
            <div className="chart-card">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold">Recent Prevention Actions</h3>
                    <Select value={incidentFilter} onValueChange={setIncidentFilter}>
                        <SelectTrigger className="w-48">
                            <SelectValue placeholder="Filter" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="all">All Events</SelectItem>
                            <SelectItem value="blocked">Blocked</SelectItem>
                            <SelectItem value="critical">Critical</SelectItem>
                            <SelectItem value="high">High Risk</SelectItem>
                            <SelectItem value="none">No Action</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
                <div className="overflow-auto max-h-96">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Time</TableHead>
                                <TableHead>Source IP</TableHead>
                                <TableHead>Prediction</TableHead>
                                <TableHead>Confidence</TableHead>
                                <TableHead>Risk</TableHead>
                                <TableHead>Action</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {filteredIncidents.length === 0 && (
                                <TableRow>
                                    <TableCell
                                        colSpan={6}
                                        className="text-center text-muted-foreground py-8"
                                    >
                                        No incidents recorded yet
                                    </TableCell>
                                </TableRow>
                            )}
                            {filteredIncidents.map((inc) => (
                                <TableRow key={inc.id}>
                                    <TableCell className="text-sm font-mono">
                                        {new Date(inc.timestamp).toLocaleString()}
                                    </TableCell>
                                    <TableCell className="font-mono text-sm">
                                        {inc.source_ip}
                                    </TableCell>
                                    <TableCell className="text-sm">{inc.prediction}</TableCell>
                                    <TableCell className="text-sm">
                                        {(inc.confidence * 100).toFixed(1)}%
                                    </TableCell>
                                    <TableCell>
                                        <span
                                            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${riskColor(
                                                inc.risk_level
                                            )}`}
                                        >
                                            {inc.risk_level}
                                        </span>
                                    </TableCell>
                                    <TableCell>
                                        <span
                                            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${actionBadge(
                                                inc.action_taken
                                            )}`}
                                        >
                                            {inc.action_taken === "none" ? "—" : inc.action_taken}
                                        </span>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </div>
            </div>
        </div>
    );
};

export default Prevention;
