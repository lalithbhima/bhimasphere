import React, { useState } from "react";
import { getExplanation } from "../lib/api";

export default function Explanations() {
  const [starId, setStarId] = useState("");
  const [exp, setExp] = useState<any>(null);

  const handleFetch = async () => {
    const res = await getExplanation(starId);
    setExp(res);
  };

  return (
    <div>
      <h2>Explanations Viewer</h2>
      <input value={starId} onChange={(e) => setStarId(e.target.value)} placeholder="Enter Star ID" />
      <button onClick={handleFetch}>Fetch</button>
      {exp && (
        <div>
          <h3>Rationale</h3>
          <p>{exp.rationale}</p>
          <h3>Top Features</h3>
          <ul>
            {exp.top_features.map((f: any, idx: number) => (
              <li key={idx}>{f.feature}: {f.contribution}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
