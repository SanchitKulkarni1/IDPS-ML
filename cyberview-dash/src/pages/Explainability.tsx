import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Info } from "lucide-react";
import { useEffect, useState } from "react";
import { getJson } from "@/lib/api";

const shapData = [
  { feature: "Service Type", impact: 0.32, direction: "positive" },
  { feature: "Source Bytes", impact: -0.28, direction: "negative" },
  { feature: "Duration", impact: 0.24, direction: "positive" },
  { feature: "Protocol", impact: -0.18, direction: "negative" },
  { feature: "Destination Bytes", impact: 0.15, direction: "positive" },
  { feature: "Flag Status", impact: -0.12, direction: "negative" },
  { feature: "Wrong Fragment", impact: 0.08, direction: "positive" },
  { feature: "Land Connection", impact: -0.05, direction: "negative" },
];

const Explainability = () => {
   const [fi, setFi] = useState<any[]>([]);
    useEffect(() => {
    (async () => {
      try {
        const res = await getJson("/api/analytics/feature-importance");
        setFi(res);
      } catch (err) {
        console.error("feature importance fetch failed", err);
      }
    })();
  }, []);
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Model Explainability (SHAP)</h1>
        <p className="text-muted-foreground mt-1">Understanding feature impact on predictions</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 chart-card">
          <h3 className="text-lg font-semibold mb-4">SHAP Feature Impact Analysis</h3>
          <p className="text-sm text-muted-foreground mb-6">
            Positive values increase malicious probability, negative values decrease it
          </p>
          <ResponsiveContainer width="100%" height={450}>
            <BarChart data={fi} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis type="number" stroke="hsl(var(--muted-foreground))" />
              <YAxis
                dataKey="feature"
                type="category"
                stroke="hsl(var(--muted-foreground))"
                width={150}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "0.5rem",
                }}
                formatter={(value: number) => [
                  `${value > 0 ? "+" : ""}${value.toFixed(2)}`,
                  "Impact",
                ]}
              />
              <Bar dataKey="impact" radius={[0, 8, 8, 0]}>
                {fi.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      entry.direction === "positive"
                        ? "hsl(var(--destructive))"
                        : "hsl(var(--success))"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="space-y-6">
          {/* Explanation Box */}
          <div className="stat-card bg-primary/5 border-primary">
            <div className="flex items-start gap-3">
              <Info className="h-5 w-5 text-primary mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-foreground mb-2">How to Read This</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  SHAP (SHapley Additive exPlanations) values show how much each feature
                  contributed to the prediction. Red bars push the prediction toward
                  "malicious", while green bars push it toward "normal".
                </p>
              </div>
            </div>
          </div>

          {/* Key Insights */}
          <div className="stat-card">
            <h3 className="font-semibold mb-4">Key Insights</h3>
            <div className="space-y-3">
              <div className="p-3 bg-destructive/10 rounded-lg border border-destructive/20">
                <p className="text-sm font-medium text-destructive mb-1">High Risk Indicators</p>
                <p className="text-xs text-muted-foreground">
                  Service Type and Duration are the strongest predictors of malicious traffic
                </p>
              </div>
              <div className="p-3 bg-success/10 rounded-lg border border-success/20">
                <p className="text-sm font-medium text-success mb-1">Safety Indicators</p>
                <p className="text-xs text-muted-foreground">
                  Normal protocol usage and low source bytes suggest legitimate traffic
                </p>
              </div>
            </div>
          </div>

          {/* Model Info */}
          <div className="stat-card">
            <h3 className="font-semibold mb-3">Model Information</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Algorithm</span>
                <span className="font-medium">Random Forest</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Accuracy</span>
                <span className="font-medium">96.8%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">F1 Score</span>
                <span className="font-medium">95.2%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Training Samples</span>
                <span className="font-medium">125,973</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Explainability;
