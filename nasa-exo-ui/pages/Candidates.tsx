import React, { useEffect, useState } from "react";
import { getCandidates } from "../lib/api";

export default function Candidates() {
  const [rows, setRows] = useState<any[]>([]);

  useEffect(() => {
    getCandidates().then(setRows);
  }, []);

  return (
    <div>
      <h2>Candidate Browser</h2>
      <table border={1} cellPadding={4} style={{ width: "100%" }}>
        <thead>
          <tr>
            <th>Star ID</th>
            <th>Prediction</th>
            <th>p_FP</th>
            <th>p_Cand</th>
            <th>p_Conf</th>
            <th>Period</th>
            <th>Duration</th>
            <th>Depth</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              <td>{row.star_id}</td>
              <td>{row.pred_text}</td>
              <td>{row.p_fp?.toFixed(3)}</td>
              <td>{row.p_cand?.toFixed(3)}</td>
              <td>{row.p_conf?.toFixed(3)}</td>
              <td>{row.period}</td>
              <td>{row.dur_h}</td>
              <td>{row.depth_ppm}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
