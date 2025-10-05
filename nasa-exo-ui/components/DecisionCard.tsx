import React from "react";

export default function DecisionCard({ result }: { result: any }) {
  return (
    <div style={{ border: "1px solid #ccc", padding: 16, marginTop: 20 }}>
      <h3>Prediction: {result.pred_text}</h3>
      <p>Probabilities:</p>
      <ul>
        <li>False Positive: {result.proba[0].toFixed(3)}</li>
        <li>Candidate: {result.proba[1].toFixed(3)}</li>
        <li>Confirmed: {result.proba[2].toFixed(3)}</li>
      </ul>
      <p>Rationale: {result.rationale}</p>
    </div>
  );
}
