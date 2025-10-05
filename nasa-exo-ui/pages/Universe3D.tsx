// nasa-exo-ui/pages/Universe3D.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
// import static exoplanets data
import { exoplanets } from "../js/data/exoplanets";
// Import planet name/ID datasets
import keplerData from "../js/data/kepler_objects_of_interest.json";
import k2Data from "../js/data/k2_planets_and_candidates.json";
import tessData from "../js/data/TESS_objects_of_interest.json";


// ---------- Types (matches your /api/universe + /api/metrics contracts) ----------
type UniverseRow = {
  mission: string;
  pred_text: string;  // from JSON
  label?: string;     // optional for API compatibility
  ra_deg: number;
  dec_deg: number;
  dist_pc?: number | null;
  period?: number | null;
  rade_Re?: number | null;
  p_exoplanet?: number | null;
};

type MetricsResponse = {
  scores?: {
    oof?: { pr_auc_macro?: number; roc_auc_ovo?: number };
    holdout?: { pr_auc_macro?: number; roc_auc_ovo?: number };
  };
  best_thresholds?: Record<string, number>; // "2" => tau Confirmed
  conformal_q?: Record<string, number>;
  weights?: Record<string, number>;
  feature_count?: number;
  reproducibility?: any;
};

type ReleaseResponse = {
  release?: string | null;
  paths?: Record<string, any>;
  reproducibility?: any;
};

// ---------- Config ----------
const API_BASE = "http://127.0.0.1:7860";
const DEFAULT_LIMIT = 5000;

// Mission colors (friendly, high-contrast)
const MISSION_COLORS: Record<string, number> = {
  Kepler: 0x60a5fa, // blue
  K2: 0xa78bfa,     // violet
  TESS: 0xf472b6,   // pink
};

// Label colors (semantic)
const LABEL_COLORS: Record<string, number> = {
  Confirmed: 0x4ade80,      // green
  Candidate: 0xfbbf24,      // yellow
  "False Positive": 0xef4444 // red
};

// Probability → color (simple turbo-like ramp)
function colorFromProb(p: number): THREE.Color {
  // clamp
  const x = Math.max(0, Math.min(1, p));
  // smooth gradient blue->cyan->yellow->orange->red
  const c = new THREE.Color();
  c.setHSL(0.66 - 0.66 * x, 0.85, 0.5 + 0.15 * (x - 0.5));
  return c;
}

// RA/Dec → unit vector
function radecToUnit(raDeg: number, decDeg: number): THREE.Vector3 {
  const ra = (raDeg || 0) * Math.PI / 180;
  const dec = (decDeg || 0) * Math.PI / 180;
  const x = Math.cos(dec) * Math.cos(ra);
  const y = Math.cos(dec) * Math.sin(ra);
  const z = Math.sin(dec);
  return new THREE.Vector3(x, y, z);
}

// Lookup planet name/ID by mission and coordinates
function getPlanetName(row: UniverseRow): string {
  let name: string | null = null;

  if (row.mission === "Kepler") {
    const match = (keplerData as any[]).find(
      (p) => Math.abs(p.koi_ra - row.ra_deg) < 0.01 && Math.abs(p.koi_dec - row.dec_deg) < 0.01
    );
    name = match?.kepoi_name || match?.kepler_name || null;
  } else if (row.mission === "K2") {
    const match = (k2Data as any[]).find(
      (p) => Math.abs(p.ra - row.ra_deg) < 0.01 && Math.abs(p.dec - row.dec_deg) < 0.01
    );
    name = match?.pl_name || match?.k2_name || match?.epic_candname || null;
  } else if (row.mission === "TESS") {
    const match = (tessData as any[]).find(
      (p) => Math.abs(p.ra - row.ra_deg) < 0.01 && Math.abs(p.dec - row.dec_deg) < 0.01
    );
    name = match?.toi ? `TOI-${match.toi}` : match?.ctoi_alias || null;
  }

  return name || "Unlabeled Object";
}


function teffToColor(teff: number | null | undefined): THREE.Color {
  if (!teff) return new THREE.Color(0xffffff);
  if (teff > 10000) return new THREE.Color(0x9bb0ff); // blue
  if (teff > 7500)  return new THREE.Color(0xaabfff); // light blue
  if (teff > 6000)  return new THREE.Color(0xf8f7ff); // white-yellow
  if (teff > 5200)  return new THREE.Color(0xfff4e8); // yellow
  if (teff > 3700)  return new THREE.Color(0xffd2a1); // orange
  return new THREE.Color(0xff6f61);                   // red dwarf
}

// ---------- Component ----------
export default function Universe3D() {
  // Scene refs
  const [showStars, setShowStars] = useState<"Yes" | "No" | "OnlyStars">("Yes");
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const controlsRef = useRef<OrbitControls>();
  const sceneRef = useRef<THREE.Scene>();
  const cloudRef = useRef<THREE.InstancedMesh | null>(null);
  const raycaster = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const starsRef = useRef<THREE.InstancedMesh | null>(null);
  const mouse = useRef(new THREE.Vector2());

  // Data
  const [rows, setRows] = useState<UniverseRow[]>([]);
  const [countShown, setCountShown] = useState<number>(0);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [release, setRelease] = useState<ReleaseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // UI state
  const [limit, setLimit] = useState<number>(DEFAULT_LIMIT);
  const [source, setSource] = useState<"auto" | "exounified" | "candidates">("exounified");
  const [colorBy, setColorBy] = useState<"mission" | "label" | "probability">("mission");
  const [missionFilter, setMissionFilter] = useState<"All" | "Kepler" | "K2" | "TESS">("All");
  const [labelFilter, setLabelFilter] = useState<"All" | "Confirmed" | "Candidate" | "False Positive">("All");
  const [probMin, setProbMin] = useState<number>(0.0);
  const [radiusMin, setRadiusMin] = useState<number>(0.0);
  const [periodMax, setPeriodMax] = useState<number | "">("");
  const [distanceMix, setDistanceMix] = useState<number>(0.5); // blend: 0 = shell, 1 = distance-coded
  const [baseSize, setBaseSize] = useState<number>(1.0);       // mesh scale multiplier
  const [picked, setPicked] = useState<{ idx: number; row: UniverseRow } | null>(null);
  // ---- Mode + retrain/upload UI state ----
  const [mode, setMode] = useState<"novice" | "researcher">("novice");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  // ---- API helpers ----
  async function fetchMetrics() {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      setMetrics(await res.json());
    } catch (e) {
      console.error(e);
    }
  }

  async function handleUpload() {
    if (!uploadFile) return alert("Please select a CSV first.");
    const formData = new FormData();
    formData.append("file", uploadFile);
    // If your backend requires token, append as query or header. For ALLOW_RETRAIN=1 you can omit.
    const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) return alert(`Upload failed: ${data?.detail ?? res.statusText}`);
    alert(`✅ Uploaded: ${data.saved_as ?? "ingested (see logs)"}`);
  }

  async function handleRetrain() {
    if (!confirm("Run full retrain? This may take a few minutes.")) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("notes", "Triggered from Researcher UI");
      const res = await fetch(`${API_BASE}/api/retrain`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail ?? "Retrain failed");
      alert(`✅ Retrain complete: Release ${data.release?.release ?? data.release ?? "(see registry/latest)"}`);
      // Pull fresh KPIs after retrain:
      fetchMetrics();
    } catch (e:any) {
      alert(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  // Learned Confirmed threshold (fallback = 0.5)
  const tauConfirmed = useMemo(() => {
    const t = metrics?.best_thresholds?.["2"];
    return Number.isFinite(t) ? Number(t) : 0.5;
  }, [metrics]);

  // ⬇️ ADD THIS EFFECT to toggle body class when Universe is open
  useEffect(() => {
    document.body.classList.add("universe-open");
    return () => document.body.classList.remove("universe-open");
  }, []);

  // ---------- Load static exoplanets.js ----------
  useEffect(() => {
    setError(null);
    // filter rows with valid RA/Dec
    const good = exoplanets.filter(
      (r) => Number.isFinite(r.ra_deg) && Number.isFinite(r.dec_deg)
    );
    console.log("Loaded exoplanets.js:", good.length, "rows");
    setRows(good);
  }, []);

  useEffect(() => {
    fetchMetrics();
    // (Optional) fetch release banner if you wire a helper later.
  }, []);  

  // ---------- Debug scroll/viewport ----------
  useEffect(() => {
    console.log("scrollY:", window.scrollY, "innerHeight:", window.innerHeight);

    const onScroll = () => {
      console.log("scrollY (live):", window.scrollY);
    };
    window.addEventListener("scroll", onScroll);

    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // ---------- Filtered view ----------
  const filtered = useMemo(() => {
    return rows.filter((r) => {
      if (missionFilter !== "All" && r.mission !== missionFilter) return false;
      if (labelFilter !== "All" && (r.pred_text || r.label) !== labelFilter) return false;
      if (r.p_exoplanet != null && r.p_exoplanet < probMin) return false;
      if (r.rade_Re != null && r.rade_Re < radiusMin) return false;
      if (periodMax !== "" && r.period != null && r.period > Number(periodMax)) return false;
      return true;
    });
  }, [rows, missionFilter, labelFilter, probMin, radiusMin, periodMax]);

  // ---------- Unique stars (one per host) ----------
  const uniqueStars = useMemo(() => {
    const seen = new Map<string, UniverseRow>();
    rows.forEach(r => {
      const id = (r as any).kepid || (r as any).hostname || `${r.ra_deg}_${r.dec_deg}`;
      if (!seen.has(id)) seen.set(id, r);
    });
    return Array.from(seen.values());
  }, [rows]);


  // ---------- Three.js: scene lifecycle ----------
  useEffect(() => {
    if (!mountRef.current) return;
    const rect = mountRef.current.getBoundingClientRect();
    console.log("Universe3D mountRef rect:", rect);
    console.log("Universe3D parent rect:", mountRef.current.parentElement?.getBoundingClientRect());
    console.log("Body rect:", document.body.getBoundingClientRect());
    console.log("#root rect:", document.getElementById("root")?.getBoundingClientRect());

    // Scene/camera/renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 20000);
    camera.position.set(0, 0, 800);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // ⬇️ ADD THIS RIGHT HERE
    const canvasEl = renderer.domElement as HTMLCanvasElement;
    Object.assign(canvasEl.style, {
      position: "fixed",   // anchor to viewport
      top: "0",
      left: "0",
      width: "100vw",
      height: "100vh",
      zIndex: "1",         // make sure it's above background layers
      display: "block",
    });

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 80;
    controls.maxDistance = 20000;
    controlsRef.current = controls;

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.6)); // global soft light
    const dir = new THREE.DirectionalLight(0xffffff, 1.0); // stronger key light
    dir.position.set(50, 50, 50);
    scene.add(dir);

    // Mouse move for picking
    const onMove = (e: MouseEvent) => {
      mouse.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.current.y = -(e.clientY / window.innerHeight) * 2 + 1;
    };
    window.addEventListener("mousemove", onMove);

    const onResize = () => {
      if (!rendererRef.current || !cameraRef.current) return;
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", onResize);

    const animate = () => {
      requestAnimationFrame(animate);
      // subtle drift for background parallax
      if (cloudRef.current) cloudRef.current.rotation.y += 0.000;
      controls.update();

      // picking
      if (cloudRef.current) {
        raycaster.current.setFromCamera(mouse.current, camera);
        const hit = raycaster.current.intersectObject(cloudRef.current, false)[0];
        (cloudRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.15;
        if (hit && typeof hit.instanceId === "number") {
          // highlight picked by slightly boosting emissive (visual hint)
          (cloudRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.25;
          const idx = hit.instanceId;
          const row = (cloudRef.current.userData.rows as UniverseRow[])[idx];
          if (row && (!picked || picked.idx !== idx)) {
            setPicked({ idx, row });
          }
        }
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      mountRef.current?.removeChild(renderer.domElement);
      cloudRef.current = null;
      sceneRef.current = undefined;
      cameraRef.current = undefined;
      rendererRef.current = undefined;
    };
  }, []); // once

  // ---------- Build/refresh planet cloud when filtered/color/size settings change ----------
  // ---------- Build/refresh planet cloud --------
  useEffect(() => {
    if (!sceneRef.current) return;

    // Always remove old planets first
    if (cloudRef.current) {
      cloudRef.current.geometry.dispose();
      (cloudRef.current.material as THREE.Material).dispose();
      sceneRef.current.remove(cloudRef.current);
      cloudRef.current = null;
    }

    if (showStars === "OnlyStars") {
      setCountShown(uniqueStars.length); // count stars instead
      return; // ⬅️ skip planets entirely
    }

    const N = filtered.length;
    setCountShown(N + (showStars === "Yes" ? uniqueStars.length : 0));
    if (!N) return;

    // Geometry: base sphere
    const geo = new THREE.SphereGeometry(30.0, 32, 32);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,            // solid white
      metalness: 0.1,
      roughness: 0.4,
      emissive: new THREE.Color(0x000000),
      emissiveIntensity: 0.0,
      vertexColors: false         // ❌ disable per-instance colors
    });

    const mesh = new THREE.InstancedMesh(geo, mat, N);
    const tmpM = new THREE.Matrix4();
    const tmpQ = new THREE.Quaternion();
    const tmpS = new THREE.Vector3(1, 1, 1);
    const tmpC = new THREE.Color();

    // Scale factor: shrink parsecs to fit inside view
    const DIST_SCALE = 15.0; // adjust this up/down to make scene fit

    for (let i = 0; i < N; i++) {
      const r = filtered[i];

      // RA/Dec → direction unit vector
      const dir = radecToUnit(r.ra_deg, r.dec_deg);

      // Use real Gaia distance (scaled down)
      const R = (r.dist_pc ?? 0) * DIST_SCALE;

      // Final Cartesian position
      const pos = dir.multiplyScalar(R);

      // Planet size = cube root of radius (Earth ~1, Jupiter ~11 → ~2.2x bigger)
      const rp = Number.isFinite(r.rade_Re ?? NaN) ? r.rade_Re : 1;
      const scale = Math.cbrt(rp) * 0.5 * baseSize;
      tmpS.set(scale, scale, scale);

      tmpM.compose(pos, tmpQ, tmpS);
      mesh.setMatrixAt(i, tmpM);

      // Debug: log first 10 planets
      if (i < 10) {
        console.log("Planet", i, {
          ra: r.ra_deg,
          dec: r.dec_deg,
          dist: r.dist_pc,
          pos: pos.toArray(),
        });
      }
      (r as any).planet_name = getPlanetName(r);
    }

    mesh.instanceMatrix.needsUpdate = true;
    mesh.userData.rows = filtered;

    sceneRef.current.add(mesh);
    cloudRef.current = mesh;
    // ✅ recenter camera/controls on cloud
    if (controlsRef.current && cameraRef.current) {
      // Compute centroid (average) of all positions
      const centroid = new THREE.Vector3();
      for (let i = 0; i < N; i++) {
        const row = filtered[i];
        const dir = radecToUnit(row.ra_deg, row.dec_deg);
        const R = (row.dist_pc ?? 0) * DIST_SCALE;
        centroid.add(dir.multiplyScalar(R));
      }
      centroid.divideScalar(N);

      // Compute spread relative to centroid
      let maxDist = 0;
      for (let i = 0; i < N; i++) {
        const row = filtered[i];
        const dir = radecToUnit(row.ra_deg, row.dec_deg);
        const R = (row.dist_pc ?? 0) * DIST_SCALE;
        const pos = dir.multiplyScalar(R);
        maxDist = Math.max(maxDist, pos.distanceTo(centroid));
      }

      // Recenter controls/camera
      controlsRef.current.target.copy(centroid);
      cameraRef.current.position.set(centroid.x, centroid.y, centroid.z + maxDist * 1.5);
      controlsRef.current.update();
    }
  }, [filtered, colorBy, baseSize, showStars]);
  // ---------- Build/refresh star mesh ----------
  useEffect(() => {
    if (!sceneRef.current) return;

    // Remove old stars
    if (starsRef.current) {
      starsRef.current.geometry.dispose();
      (starsRef.current.material as THREE.Material).dispose();
      sceneRef.current.remove(starsRef.current);
      starsRef.current = null;
    }
  if (showStars === "No") return; // Skip rendering stars

    const N = uniqueStars.length;
    if (!N) return;

    // Create geometry
    const geo = new THREE.SphereGeometry(10.0, 64, 64);

    // ⭐ Procedural canvas texture for star surface
    function makeStarTexture() {
      const size = 256;
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d")!;
      const imageData = ctx.createImageData(size, size);
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const i = (y * size + x) * 4;

          // Simple noise: combine x/y sinusoids
          const v =
            0.5 +
            0.25 * Math.sin(x * 0.1) +
            0.25 * Math.cos(y * 0.1 + x * 0.05);

          // Base yellow (255, 215, 0) with variation
          const base = [255, 215, 0];
          const light = [255, 255, 153]; // lighter yellow
          const dark = [230, 184, 0];    // darker yellow

          // Interpolate based on v
          const r = base[0] * (1 - v) + light[0] * v;
          const g = base[1] * (1 - v) + light[1] * v;
          const b = base[2] * (1 - v) + light[2] * v;

          imageData.data[i + 0] = r;
          imageData.data[i + 1] = g;
          imageData.data[i + 2] = b;
          imageData.data[i + 3] = 255;
        }
      }
      ctx.putImageData(imageData, 0, 0);
      const texture = new THREE.CanvasTexture(canvas);
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
      texture.anisotropy = 8;
      return texture;
    }

    const starTexture = makeStarTexture();

    // Material with emissive glow
    const mat = new THREE.MeshStandardMaterial({
      map: starTexture,
      emissive: new THREE.Color(0xffdd33),
      emissiveIntensity: 2.0,
      roughness: 0.6,
      metalness: 0.0,
    });

    const mesh = new THREE.InstancedMesh(geo, mat, N);

    const tmpM = new THREE.Matrix4();
    const tmpQ = new THREE.Quaternion();
    const tmpS = new THREE.Vector3(1, 1, 1);

    const DIST_SCALE = 10.0;

    for (let i = 0; i < N; i++) {
      const star = uniqueStars[i];
      const dir = radecToUnit(star.ra_deg, star.dec_deg);
      const R = (star.dist_pc ?? 0) * DIST_SCALE;
      const pos = dir.multiplyScalar(R);

      // Size ∝ stellar radius
      const sr = (star as any).st_rad ?? 1;
      const scale = Math.cbrt(sr) * 0.3;
      tmpS.set(scale, scale, scale);

      tmpM.compose(pos, tmpQ, tmpS);
      mesh.setMatrixAt(i, tmpM);
    }

    mesh.instanceMatrix.needsUpdate = true;

    sceneRef.current.add(mesh);
    starsRef.current = mesh;
  }, [uniqueStars, showStars]);

  // ---------- Actions ----------
  const resetView = () => {
    if (!controlsRef.current || !cameraRef.current || !cloudRef.current) return;

    // Compute bounding box of the current instanced mesh
    const box = new THREE.Box3().setFromObject(cloudRef.current);
    const center = new THREE.Vector3();
    box.getCenter(center);

    // Move camera behind the cloud, looking at the center
    controlsRef.current.target.copy(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);

    cameraRef.current.position.set(center.x, center.y, center.z + maxDim * 1.5);
    controlsRef.current.update();
    controlsRef.current.update();
  };

  const screenshot = () => {
    const r = rendererRef.current;
    if (!r) return;
    const url = r.domElement.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = "universe.png";
    a.click();
  };

  const exportCSV = () => {
    const cols = [
      "mission", "label", "ra_deg", "dec_deg", "dist_pc",
      "period", "rade_Re", "depth_ppm", "p_exoplanet"
    ];
    const lines = [cols.join(",")];
    filtered.forEach((r) => {
      lines.push(cols.map((c) => (r as any)[c] ?? "").join(","));
    });
    const csv = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(csv);
    const a = document.createElement("a");
    a.href = url;
    a.download = "universe_filtered.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  // ---------- Small helpers ----------
  const pill = (txt: React.ReactNode) => (
    <span style={{
      background: "rgba(2,6,23,.75)",
      border: "1px solid rgba(226,232,240,.15)",
      padding: "5px 9px",
      borderRadius: 999,
      fontSize: 12,
      color: "#cbd5e1",
      marginLeft: 6
    }}>{txt}</span>
  );

  // ---------- Render ----------
  return (
    <div style={{ width: "100%", height: "100vh", margin: 0, padding: 0, position: "absolute", top: 0, left: 0 }}>
      <div
        ref={mountRef}
        style={{
          width: "100%",
          height: "100vh",   // full viewport height
          margin: 0,
          padding: 0,
          display: "block",
          position: "absolute",
          top: 0,
          left: 0
        }}
      />

      {/* Top-left Controls */}
      <div style={{
        position: "fixed", left: 12, top: 12, zIndex: 10,
        display: "flex", gap: 12, alignItems: "center"
      }}>
        <div style={{
          background: "rgba(2,6,23,.85)",
          border: "1px solid rgba(226,232,240,.15)",
          padding: 0, borderRadius: 12, color: "#e2e8f0", minWidth: 300
        }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
            <b>3D Universe</b>
            {pill(`${countShown.toLocaleString()} objects`)}
            {pill(`τ₍Confirmed₎ ≈ ${tauConfirmed.toFixed(3)}`)}
          </div>

          {/* Mode toggle */}
          <div style={{ display: "flex", gap: 6, margin: "0 0 6px 0" }}>
            <button
              onClick={() => setMode("novice")}
              style={{ ...btnStyle, padding: "4px 10px", opacity: mode === "novice" ? 1 : 0.6 }}
              aria-pressed={mode === "novice"}
            >
              Novice Mode
            </button>
            <button
              onClick={() => setMode("researcher")}
              style={{ ...btnStyle, padding: "4px 10px", opacity: mode === "researcher" ? 1 : 0.6 }}
              aria-pressed={mode === "researcher"}
            >
              Research Mode
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Source
              <select
                value={source}
                onChange={(e) => setSource(e.target.value as any)}
                style={{ width: "100%", marginTop: 4, background: "#0b1220", color: "#e2e8f0", borderRadius: 8, border: "1px solid #22324d", padding: "6px 8px" }}>
                <option value="exounified">Processed</option>
                <option value="candidates">Candidates</option>
                <option value="auto">Auto</option>
              </select>
            </label>

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Show Stars
              <select
                value={showStars}
                onChange={(e) => setShowStars(e.target.value as "Yes" | "No" | "OnlyStars")}
                style={{
                  width: "100%",
                  marginTop: 4,
                  background: "#0b1220",
                  color: "#e2e8f0",
                  borderRadius: 8,
                  border: "1px solid #22324d",
                  padding: "6px 8px"
                }}
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="OnlyStars">Only Stars</option>
              </select>
            </label>            

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Limit
              <input
                type="number" min={100} max={50000} step={100}
                value={limit}
                onChange={(e) => setLimit(parseInt(e.target.value || "1000"))}
                style={{ width: "100%", marginTop: 4, background: "#0b1220", color: "#e2e8f0", borderRadius: 8, border: "1px solid #22324d", padding: "6px 8px" }}
              />
            </label>

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Mission
              <select
                value={missionFilter}
                onChange={(e) => setMissionFilter(e.target.value as any)}
                style={{ width: "100%", marginTop: 4, background: "#0b1220", color: "#e2e8f0", borderRadius: 8, border: "1px solid #22324d", padding: "6px 8px" }}>
                <option>All</option>
                <option>Kepler</option>
                <option>K2</option>
                <option>TESS</option>
              </select>
            </label>

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Label
              <select
                value={labelFilter}
                onChange={(e) => setLabelFilter(e.target.value as any)}
                style={{ width: "100%", marginTop: 4, background: "#0b1220", color: "#e2e8f0", borderRadius: 8, border: "1px solid #22324d", padding: "6px 8px" }}>
                <option>All</option>
                <option>Confirmed</option>
                <option>Candidate</option>
                <option>False Positive</option>
              </select>
            </label>

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Min p(exoplanet)
              <input
                type="range" min={0} max={1} step={0.01}
                value={probMin}
                onChange={(e) => setProbMin(parseFloat(e.target.value))}
                style={{ width: "100%", marginTop: 4 }}
              />
              <div style={{ fontSize: 12 }}>≥ {probMin.toFixed(2)}</div>
            </label>

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Min Radius (R⊕)
              <input
                type="number" min={0} max={50} step={0.1}
                value={radiusMin}
                onChange={(e) => setRadiusMin(parseFloat(e.target.value || "0"))}
                style={{ width: "100%", marginTop: 4, background: "#0b1220", color: "#e2e8f0", borderRadius: 8, border: "1px solid #22324d", padding: "6px 8px" }}
              />
            </label>

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Max Period (days)
              <input
                type="number" min={0} step={0.1}
                value={periodMax}
                onChange={(e) => setPeriodMax(e.target.value === "" ? "" : parseFloat(e.target.value))}
                style={{ width: "100%", marginTop: 4, background: "#0b1220", color: "#e2e8f0", borderRadius: 8, border: "1px solid #22324d", padding: "6px 8px" }}
              />
            </label>

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Color By
              <select
                value={colorBy}
                onChange={(e) => setColorBy(e.target.value as any)}
                style={{ width: "100%", marginTop: 4, background: "#0b1220", color: "#e2e8f0", borderRadius: 8, border: "1px solid #22324d", padding: "6px 8px" }}>
                <option value="mission">Mission</option>
                <option value="label">Label</option>
                <option value="probability">Probability</option>
              </select>
            </label>

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Distance Mix
              <input
                type="range" min={0} max={1} step={0.01}
                value={distanceMix}
                onChange={(e) => setDistanceMix(parseFloat(e.target.value))}
                style={{ width: "100%", marginTop: 4 }}
              />
              <div style={{ fontSize: 12 }}>{(distanceMix * 100).toFixed(0)}% distance-coded</div>
            </label>

            <label style={{ fontSize: 12, color: "#94a3b8" }}>
              Base Size
              <input
                type="range" min={0.5} max={3} step={0.05}
                value={baseSize}
                onChange={(e) => setBaseSize(parseFloat(e.target.value))}
                style={{ width: "100%", marginTop: 4 }}
              />
              <div style={{ fontSize: 12 }}>×{baseSize.toFixed(2)}</div>
            </label>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 10 }}>
            {/* Always-visible actions */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button onClick={resetView} style={btnStyle}>Reset View</button>
              <button onClick={screenshot} style={btnStyle}>Screenshot</button>
              <button onClick={exportCSV} style={btnStyle}>Export CSV</button>
            </div>

            {/* Research-only: upload + retrain */}
            {mode === "researcher" && (
              <div
                style={{
                  marginTop: 6,
                  padding: 8,
                  borderRadius: 10,
                  border: "1px solid #22324d",
                  background: "rgba(11,18,32,.6)",
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto",
                  gap: 8,
                  alignItems: "center",
                }}
                aria-label="Upload & Retrain"
              >
                {/* Sample format download */}
                <button
                  onClick={() => {
                    const sample = [
                      "mission,ra_deg,dec_deg,dist_pc,period,rade_Re,depth_ppm,label,p_exoplanet",
                      "Kepler,294.1073,46.2095,250,1.247,2.1,500,Confirmed,0.95",
                      "K2,126.5482,-14.2098,130,3.612,1.4,220,Candidate,0.72",
                      "TESS,23.5187,-72.3152,80,0.991,1.0,180,False Positive,0.08"
                    ].join("\n");

                    const blob = new Blob([sample], { type: "text/csv" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "sample_retrain_format.csv";
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  style={btnStyle}
                >
                  Sample CSV
                </button>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
                  style={{ color: "#e2e8f0" }}
                  aria-label="Upload CSV of labeled/unlabeled targets"
                />
                <button onClick={handleUpload} style={btnStyle} disabled={!uploadFile || loading}>
                  {loading ? "Working..." : "Upload CSV"}
                </button>
                <button onClick={handleRetrain} style={btnStyle} disabled={loading}>
                  {loading ? "Retraining…" : "Retrain"}
                </button>
                <div style={{ gridColumn: "1 / -1", fontSize: 12, color: "#94a3b8" }}>
                  Tip: Upload KOI/TOI/mission-like tables. Retrain runs Step-4 snapshot (and refreshes thresholds/calibration).
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right info card (selection or error) */}
      <div style={{
        position: "fixed", right: 12, top: 12, zIndex: 10,
        background: "rgba(2,6,23,.85)",
        border: "1px solid rgba(226,232,240,.15)",
        padding: 12, borderRadius: 12, color: "#e2e8f0", width: 330
      }}>
        {!error ? (
          picked ? (
            <>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>
                {picked.row.mission} • {picked.row.pred_text || picked.row.label}
              </div>
              <div style={{ fontSize: 13, color: "#60a5fa", marginBottom: 6 }}>
                { (picked.row as any).planet_name ? `Planet: ${(picked.row as any).planet_name}` : "Planet: —" }
              </div>
              <div style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.4 }}>
                <div>RA {picked.row.ra_deg.toFixed(4)}°, Dec {picked.row.dec_deg.toFixed(4)}°</div>
                <div>p(exoplanet): <b>{(picked.row.p_exoplanet ?? NaN)?.toFixed?.(3) ?? "—"}</b> {picked.row.p_exoplanet != null && (
                  (picked.row.p_exoplanet >= tauConfirmed) ? <span style={{ color: "#4ade80" }}>✓ above τ</span> : <span style={{ color: "#ef4444" }}>below τ</span>
                )}</div>
                <div>Period: {picked.row.period != null ? `${picked.row.period.toFixed(3)} d` : "—"}</div>
                <div>Radius: {picked.row.rade_Re != null ? `${picked.row.rade_Re.toFixed(2)} R⊕` : "—"}</div>
                <div>Depth: {picked.row.depth_ppm != null ? `${Math.round(picked.row.depth_ppm)} ppm` : "—"}</div>
                <div>Distance: {Number.isFinite(picked.row.dist_pc ?? NaN) ? `${Math.round(picked.row.dist_pc!)} pc` : "—"}</div>
                <div style={{ marginTop: 8, fontSize: 12, color: "#94a3b8" }}>
                  Tip: change “Color By” to <b>Probability</b> to visually rank confidence.
                </div>
              </div>
              {/* Release metrics embedded under planet info */}
              {!error && (
                <div style={{
                  marginTop: 12,
                  borderTop: "1px solid rgba(148,163,184,.2)",
                  paddingTop: 10,
                  fontSize: 12,
                  color: "#cbd5e1"
                }}>
                  <div style={{ fontSize: 12, marginBottom: 6, color: "#93c5fd" }}>
                    {release?.release ? `Release: ${release.release}` : "Release: (demo)"}
                  </div>
                  <div>
                    OOF PR-AUC (macro): <b>{metrics?.scores?.oof?.pr_auc_macro?.toFixed?.(3) ?? "—"}</b><br />
                    Holdout PR-AUC: <b>{metrics?.scores?.holdout?.pr_auc_macro?.toFixed?.(3) ?? "—"}</b><br />
                    ROC-AUC (OVO): <b>{metrics?.scores?.holdout?.roc_auc_ovo?.toFixed?.(3) ?? metrics?.scores?.oof?.roc_auc_ovo?.toFixed?.(3) ?? "—"}</b><br />
                    Best τ(Confirmed): <b>{tauConfirmed.toFixed(3)}</b>
                  </div>
                </div>
              )}              
            </>
          ) : (
            <div style={{ fontSize: 13, color: "#cbd5e1" }}>
              {mode === "novice"
                ? "Novice Mode: guided exploration. Try hovering and changing Color By → Probability."
                : "Research Mode: use Upload & Retrain to ingest CSVs and refresh KPIs."}
              
              <div style={{ marginTop: 8 }}>
                Hover over a point to inspect. Click & drag to orbit. Mousewheel to zoom.
              </div>

              <div style={{ marginTop: 8, fontSize: 12, color: "#94a3b8" }}>
                Filters and τ come from your calibrated, physics-aware classifier.
              </div>
            </div>
          )
        ) : (
          <div style={{ color: "#fca5a5", fontSize: 13 }}>
            ⚠️ {error}
            <div style={{ color: "#94a3b8", marginTop: 6, fontSize: 12 }}>
              Make sure the API is running: <code>uvicorn main:app --reload --host 127.0.0.1 --port 7860</code>
              <br />
              And try: <code>curl {API_BASE}/api/universe?source=exounified&limit=3</code>
            </div>
          </div>
        )}
      </div>

      {/* Footer legend */}
      <div style={{
        position: "fixed", bottom: 10, left: "50%", transform: "translateX(-50%)",
        color: "#94a3b8", fontSize: 12, background: "rgba(2,6,23,.7)",
        border: "1px solid rgba(226,232,240,.12)", padding: "6px 10px", borderRadius: 10
      }}>
        {colorBy === "mission" && (
          <span>
            <Swatch hex={MISSION_COLORS.Kepler} /> Kepler
            <Swatch hex={MISSION_COLORS.K2} /> K2
            <Swatch hex={MISSION_COLORS.TESS} /> TESS
          </span>
        )}
        {colorBy === "label" && (
          <span>
            <Swatch hex={LABEL_COLORS["Confirmed"]} /> Confirmed
            <Swatch hex={LABEL_COLORS["Candidate"]} /> Candidate
            <Swatch hex={LABEL_COLORS["False Positive"]} /> False Positive
          </span>
        )}
        {colorBy === "probability" && (
          <span>Probability ramp: blue (low) → red (high)</span>
        )}
      </div>
    </div>
  );
}

// ---------- Tiny UI helpers ----------
const btnStyle: React.CSSProperties = {
  background: "#0b1220",
  color: "#e2e8f0",
  border: "1px solid #22324d",
  borderRadius: 10,
  padding: "6px 10px",
  cursor: "pointer",
};

function Swatch({ hex }: { hex: number }) {
  return (
    <span style={{
      display: "inline-block", width: 12, height: 12, borderRadius: 3, margin: "0 6px 0 12px",
      background: "#" + hex.toString(16).padStart(6, "0"), border: "1px solid rgba(255,255,255,.4)"
    }} />
  );
}
