import React, { useEffect, useState } from "react";
import { getMetrics } from "../lib/api";

export default function Dashboard() {
  const [metrics, setMetrics] = useState<any>(null);

  useEffect(() => {
    getMetrics().then(setMetrics);
  }, []);

  if (!metrics) return <p>Loading metrics...</p>;

  return (
    <div>
      <h2>Dashboard</h2>
      <p>PR-AUC (macro): {metrics.pr_auc_macro}</p>
      <p>ROC-AUC (OVO): {metrics.roc_auc_ovo}</p>
      <p>ECE: {metrics.ece}</p>
      <h3>Per Mission</h3>
      <ul>
        {Object.entries(metrics.per_mission || {}).map(([mission, auc]: any) => (
          <li key={mission}>
            {mission}: PR-AUC = {auc}
          </li>
        ))}
      </ul>
    </div>
  );
}
