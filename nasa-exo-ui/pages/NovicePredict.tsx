import React, { useState } from "react";
import { predict } from "../lib/api";
import DecisionCard from "../components/DecisionCard";

export default function NovicePredict() {
  const [form, setForm] = useState({ period: 10, dur_h: 5, depth_ppm: 500 });
  const [result, setResult] = useState<any>(null);

  const handleSubmit = async () => {
    try {
      const res = await predict(form);
      setResult(res);
    } catch (err) {
      console.error("Prediction error:", err);
    }
  };

  return (
    <div style={{ color: "white", padding: "2rem", maxWidth: "600px" }}>
      <h2>🛰️ Novice Mode Prediction</h2>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <input
          type="number"
          value={form.period}
          onChange={(e) => setForm({ ...form, period: +e.target.value })}
          placeholder="Orbital Period (days)"
        />
        <input
          type="number"
          value={form.dur_h}
          onChange={(e) => setForm({ ...form, dur_h: +e.target.value })}
          placeholder="Transit Duration (hours)"
        />
        <input
          type="number"
          value={form.depth_ppm}
          onChange={(e) => setForm({ ...form, depth_ppm: +e.target.value })}
          placeholder="Transit Depth (ppm)"
        />

        <button
          onClick={handleSubmit}
          style={{ padding: "0.5rem", cursor: "pointer" }}
        >
          Predict
        </button>
      </div>

      {result && (
        <div style={{ marginTop: "2rem" }}>
          <DecisionCard result={result} />
        </div>
      )}
    </div>
  );
}
