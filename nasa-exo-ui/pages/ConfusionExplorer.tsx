import React, { useEffect, useRef } from "react";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

export default function ConfusionExplorer() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (canvasRef.current) {
      new Chart(canvasRef.current, {
        type: "bar",
        data: {
          labels: ["False Positive", "Candidate", "Confirmed"],
          datasets: [
            {
              label: "Sample Confusion Counts",
              data: [100, 200, 150], // replace with API data later
              backgroundColor: ["#ff4d4f", "#faad14", "#52c41a"],
            },
          ],
        },
      });
    }
  }, []);

  return (
    <div style={{ width: "600px", height: "400px" }}>
      <h2>Confusion Matrix Explorer</h2>
      <canvas ref={canvasRef}></canvas>
    </div>
  );
}
