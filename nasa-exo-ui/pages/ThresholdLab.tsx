import React, { useState } from "react";

export default function ThresholdLab() {
  const [thresholds, setThresholds] = useState({
    fp: 0.44,
    cand: 0.36,
    conf: 0.42,
  });

  return (
    <div>
      <h2>Threshold Lab</h2>
      {Object.entries(thresholds).map(([key, val]) => (
        <div key={key}>
          <label>{key.toUpperCase()} Threshold</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={val}
            onChange={(e) =>
              setThresholds({ ...thresholds, [key]: parseFloat(e.target.value) })
            }
          />
          {val.toFixed(2)}
        </div>
      ))}
    </div>
  );
}
