import { Routes, Route, Link, useLocation } from "react-router-dom";

import NovicePredict from "../pages/NovicePredict"; 
import Universe3D from "../pages/Universe3D";

export default function App() {
  const location = useLocation();

  return (
    <div style={{ height: "100vh", width: "100vw" }}>
      {/* Hide nav on Universe3D page */}
      {location.pathname !== "/universe" && (
        <nav style={{ padding: "1rem", background: "#111" }}>
          <Link to="/novice" style={{ color: "white", marginRight: "1rem" }}>
            Novice Mode
          </Link>
          <Link to="/universe" style={{ color: "white", marginRight: "1rem" }}>
            3D Universe
          </Link>
        </nav>
      )}

      <Routes>
        <Route
          path="/"
          element={<h1 style={{ color: "white" }}>🚀 NASA Exo App is working!</h1>}
        />
        <Route path="/novice" element={<NovicePredict />} />
        <Route path="/universe" element={<Universe3D />} />
      </Routes>
    </div>
  );
}
