import React, { useState } from "react";

export default function EnsembleLab() {
  const [weights, setWeights] = useState({
    lgbm: 0.27,
    xgb: 0.27,
    cat: 0.26,
    tab: 0.19,
  });

  return (
    <div>
      <h2>Ensemble Lab</h2>
      {Object.entries(weights).map(([key, val]) => (
        <div key={key}>
          <label>{key.toUpperCase()} Weight</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={val}
            onChange={(e) =>
              setWeights({ ...weights, [key]: parseFloat(e.target.value) })
            }
          />
          {val.toFixed(2)}
        </div>
      ))}
    </div>
  );
}
